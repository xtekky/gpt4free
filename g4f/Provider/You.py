from __future__ import annotations

import re
import json
import base64
import uuid

from ..typing import AsyncResult, Messages, ImageType, Cookies
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..image import ImageResponse, ImagePreview, EXTENSIONS_MAP, to_bytes, is_accepted_format
from ..requests import StreamSession, FormData, raise_for_status
from .you.har_file import get_telemetry_ids
from .. import debug

class You(AsyncGeneratorProvider, ProviderModelMixin):
    label = "You.com"
    url = "https://you.com"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    default_model = "gpt-4o-mini"
    default_vision_model = "agent"
    image_models = ["dall-e"]
    models = [
        default_model,
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4",
        "claude-3.5-sonnet",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "claude-2",
        "llama-3.1-70b",
        "llama-3",
        "gemini-1-5-flash",
        "gemini-1-5-pro",
        "gemini-1-0-pro",
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
        async with StreamSession(
            proxy=proxy,
            impersonate="chrome",
            timeout=(30, timeout)
        ) as session:
            cookies = await cls.get_cookies(session) if chat_mode != "default" else None
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
            params = {
                "userFiles": upload,
                "selectedChatMode": chat_mode,
            }
            if chat_mode == "custom":
                if debug.logging:
                    print(f"You model: {model}")
                params["selectedAiModel"] = model.replace("-", "_")

            async with (session.post if chat_mode == "default" else session.get)(
                f"{cls.url}/api/streamingSearch",
                data=data if chat_mode == "default" else None,
                params=params if chat_mode == "default" else data,
                headers=headers,
                cookies=cookies
            ) as response:
                await raise_for_status(response)
                async for line in response.iter_lines():
                    if line.startswith(b'event: '):
                        event = line[7:].decode()
                    elif line.startswith(b'data: '):
                        if event in ["youChatUpdate", "youChatToken"]:
                            data = json.loads(line[6:])
                        if event == "youChatToken" and event in data and data[event]:
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

    @classmethod
    async def get_cookies(cls, client: StreamSession) -> Cookies:
        if not cls._cookies or cls._cookies_used >= 5:
            cls._cookies = await cls.create_cookies(client)
            cls._cookies_used = 0
        cls._cookies_used += 1
        return cls._cookies        

    @classmethod
    def get_sdk(cls) -> str:
        return base64.standard_b64encode(json.dumps({
            "event_id":f"event-id-{str(uuid.uuid4())}",
            "app_session_id":f"app-session-id-{str(uuid.uuid4())}",
            "persistent_id":f"persistent-id-{uuid.uuid4()}",
            "client_sent_at":"","timezone":"",
            "stytch_user_id":f"user-live-{uuid.uuid4()}",
            "stytch_session_id":f"session-live-{uuid.uuid4()}",
            "app":{"identifier":"you.com"},
            "sdk":{"identifier":"Stytch.js Javascript SDK","version":"3.3.0"
        }}).encode()).decode()

    def get_auth() -> str:
        auth_uuid = "507a52ad-7e69-496b-aee0-1c9863c7c819"
        auth_token = f"public-token-live-{auth_uuid}:public-token-live-{auth_uuid}"
        auth = base64.standard_b64encode(auth_token.encode()).decode()
        return f"Basic {auth}"

    @classmethod
    async def create_cookies(cls, client: StreamSession) -> Cookies:
        if not cls._telemetry_ids:
            cls._telemetry_ids = await get_telemetry_ids()
        user_uuid = str(uuid.uuid4())
        telemetry_id = cls._telemetry_ids.pop()
        if debug.logging:
            print(f"Use telemetry_id: {telemetry_id}")
        async with client.post(
            "https://web.stytch.com/sdk/v1/passwords",
            headers={
                "Authorization": cls.get_auth(),
                "X-SDK-Client": cls.get_sdk(),
                "X-SDK-Parent-Host": cls.url,
                "Origin": "https://you.com",
                "Referer": "https://you.com/"
            },
            json={
                "dfp_telemetry_id": telemetry_id,
                "email": f"{user_uuid}@gmail.com",
                "password": f"{user_uuid}#{user_uuid}",
                "session_duration_minutes": 129600
            }
        ) as response:
            await raise_for_status(response)
            session = (await response.json())["data"]

        return {
            "stytch_session": session["session_token"],
            'stytch_session_jwt': session["session_jwt"],
            'ydc_stytch_session': session["session_token"],
            'ydc_stytch_session_jwt': session["session_jwt"],
        }
