from __future__ import annotations

import re
import json
import uuid

from ..typing import AsyncResult, Messages, ImageType, Cookies
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..image import ImageResponse, ImagePreview, EXTENSIONS_MAP, to_bytes, is_accepted_format
from ..requests import StreamSession, FormData, raise_for_status, get_nodriver
from ..cookies import get_cookies
from ..errors import MissingRequirementsError, ResponseError
from .. import debug

class You(AsyncGeneratorProvider, ProviderModelMixin):
    label = "You.com"
    url = "https://you.com"
    working = True
    default_model = "gpt-4o-mini"
    default_vision_model = "agent"
    image_models = ["dall-e"]
    models = [
        default_model,
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "grok-2",
        "claude-3.5-sonnet",
        "claude-3.5-haiku",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "llama-3.3-70b",
        "llama-3.1-70b",
        "llama-3",
        "gemini-1-5-flash",
        "gemini-1-5-pro",
        "databricks-dbrx-instruct",
        "command-r",
        "command-r-plus",
        "dolphin-2.5",
        default_vision_model,
        *image_models
    ]
    _cookies = None
    _cookies_used = 0
    _telemetry_ids = []

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        image: ImageType = None,
        image_name: str = None,
        proxy: str = None,
        timeout: int = 240,
        chat_mode: str = "default",
        cookies: Cookies = None,
        **kwargs,
    ) -> AsyncResult:
        if image is not None or model == cls.default_vision_model:
            chat_mode = "agent"
        elif not model or model == cls.default_model:
            ...
        elif model.startswith("dall-e"):
            chat_mode = "create"
            messages = [messages[-1]]
        else:
            chat_mode = "custom"
            model = cls.get_model(model)
        if cookies is None and chat_mode != "default":
            try:
                cookies = get_cookies(".you.com")
            except MissingRequirementsError:
                pass
            if not cookies or "afUserId" not in cookies:
                browser, stop_browser = await get_nodriver(proxy=proxy)
                try:
                    page = await browser.get(cls.url)
                    await page.wait_for('[data-testid="user-profile-button"]', timeout=900)
                    cookies = {}
                    for c in await page.send(nodriver.cdp.network.get_cookies([cls.url])):
                        cookies[c.name] = c.value
                    await page.close()
                finally:
                    stop_browser()
        async with StreamSession(
            proxy=proxy,
            impersonate="chrome",
            timeout=(30, timeout)
        ) as session:
            upload = ""
            if image is not None:
                upload_file = await cls.upload_file(
                    session, cookies,
                    to_bytes(image), image_name
                )
                upload = json.dumps([upload_file])
            headers = {
                "Accept": "text/event-stream",
                "Referer": f"{cls.url}/search?fromSearchBar=true&tbm=youchat",
            }
            data = {
                "userFiles": upload,
                "q": format_prompt(messages),
                "domain": "youchat",
                "selectedChatMode": chat_mode,
                "conversationTurnId": str(uuid.uuid4()),
                "chatId": str(uuid.uuid4()),
            }
            if chat_mode == "custom":
                if debug.logging:
                    print(f"You model: {model}")
                data["selectedAiModel"] = model.replace("-", "_")

            async with session.get(
                f"{cls.url}/api/streamingSearch",
                params=data,
                headers=headers,
                cookies=cookies
            ) as response:
                await raise_for_status(response)
                async for line in response.iter_lines():
                    if line.startswith(b'event: '):
                        event = line[7:].decode()
                    elif line.startswith(b'data: '):
                        if event == "error":
                            raise ResponseError(line[6:])
                        if event in ["youChatUpdate", "youChatToken"]:
                            data = json.loads(line[6:])
                        if event == "youChatToken" and event in data and data[event]:
                            if data[event].startswith("#### You\'ve hit your free quota for the Model Agent. For more usage of the Model Agent, learn more at:"):
                                continue
                            yield data[event]
                        elif event == "youChatUpdate" and "t" in data and data["t"]:
                            if chat_mode == "create":
                                match = re.search(r"!\[(.+?)\]\((.+?)\)", data["t"])
                                if match:
                                    if match.group(1) == "fig":
                                        yield ImagePreview(match.group(2), messages[-1]["content"])
                                    else:
                                        yield ImageResponse(match.group(2), match.group(1))
                                else:
                                    yield data["t"]
                            else:
                                yield data["t"]

    @classmethod
    async def upload_file(cls, client: StreamSession, cookies: Cookies, file: bytes, filename: str = None) -> dict:
        async with client.get(
            f"{cls.url}/api/get_nonce",
            cookies=cookies,
        ) as response:
            await raise_for_status(response)
            upload_nonce = await response.text()
        data = FormData()
        content_type = is_accepted_format(file)
        filename = f"image.{EXTENSIONS_MAP[content_type]}" if filename is None else filename
        data.add_field('file', file, content_type=content_type, filename=filename)
        async with client.post(
            f"{cls.url}/api/upload",
            data=data,
            headers={
                "X-Upload-Nonce": upload_nonce,
            },
            cookies=cookies
        ) as response:
            await raise_for_status(response)
            result = await response.json()
        result["user_filename"] = filename
        result["size"] = len(file)
        return result