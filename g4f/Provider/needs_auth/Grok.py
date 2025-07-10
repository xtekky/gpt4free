from __future__ import annotations

import os
import json
import time
import asyncio
import uuid
from typing import Dict, Any, AsyncIterator

try:
    import nodriver
except ImportError:
    pass

from ...typing import Messages, AsyncResult
from ...providers.response import JsonConversation, Reasoning, ImagePreview, ImageResponse, TitleGeneration, AuthResult, RequestLogin
from ...requests import StreamSession, get_nodriver, DEFAULT_HEADERS, merge_cookies
from ...requests.raise_for_status import raise_for_status
from ...errors import MissingAuthError
from ..base_provider import AsyncAuthedProvider, ProviderModelMixin
from ..helper import format_prompt, get_last_user_message

class Conversation(JsonConversation):
    def __init__(self,
        conversation_id: str
    ) -> None:
        self.conversation_id = conversation_id

class Grok(AsyncAuthedProvider, ProviderModelMixin):
    label = "Grok AI"
    url = "https://grok.com"
    cookie_domain = ".grok.com"
    assets_url = "https://assets.grok.com"
    conversation_url = "https://grok.com/rest/app-chat/conversations"

    needs_auth = True
    working = True

    default_model = "grok-3"
    models = [default_model, "grok-3-thinking", "grok-2"]
    model_aliases = {"grok-3-r1": "grok-3-thinking"}

    @classmethod
    async def on_auth_async(cls, proxy: str = None, **kwargs) -> AsyncIterator:
        auth_result = AuthResult(headers=DEFAULT_HEADERS, impersonate="chrome")
        auth_result.headers["referer"] = cls.url + "/"
        browser, stop_browser = await get_nodriver(proxy=proxy)
        yield RequestLogin(cls.__name__, os.environ.get("G4F_LOGIN_URL") or "")
        try:
            page = await browser.get(cls.url)
            has_headers = False
            def on_request(event: nodriver.cdp.network.RequestWillBeSent, page=None):
                nonlocal has_headers
                if event.request.url.startswith(cls.conversation_url + "/new"):
                    for key, value in event.request.headers.items():
                        auth_result.headers[key.lower()] = value
                    has_headers = True
            await page.send(nodriver.cdp.network.enable())
            page.add_handler(nodriver.cdp.network.RequestWillBeSent, on_request)
            await page.reload()
            auth_result.headers["user-agent"] = await page.evaluate("window.navigator.userAgent", return_by_value=True)
            while True:
                if has_headers:
                    break
                await asyncio.sleep(1)
            auth_result.cookies = {}
            for c in await page.send(nodriver.cdp.network.get_cookies([cls.url])):
                auth_result.cookies[c.name] = c.value
            await page.close()
        finally:
            stop_browser()
        yield auth_result

    @classmethod
    async def _prepare_payload(cls, model: str, message: str) -> Dict[str, Any]:
        return {
            "temporary": True,
            "modelName": "grok-latest" if model == "grok-2" else "grok-3",
            "message": message,
            "fileAttachments": [],
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": {},
            "enableSideBySide": True,
            "isPreset": False,
            "sendFinalMetadata": True,
            "customInstructions": "",
            "deepsearchPreset": "",
            "isReasoning": model.endswith("-thinking") or model.endswith("-r1"),
        }

    @classmethod
    async def create_authed(
        cls,
        model: str,
        messages: Messages,
        auth_result: AuthResult,
        conversation: Conversation = None,
        **kwargs
    ) -> AsyncResult:
        conversation_id = None if conversation is None else conversation.conversation_id
        prompt = format_prompt(messages) if conversation_id is None else get_last_user_message(messages)
        async with StreamSession(
            **auth_result.get_dict()
        ) as session:
            payload = await cls._prepare_payload(model, prompt)
            if conversation_id is None:
                url = f"{cls.conversation_url}/new"
            else:
                url = f"{cls.conversation_url}/{conversation_id}/responses"
            async with session.post(url, json=payload, headers={"x-xai-request-id": str(uuid.uuid4())}) as response:
                if response.status == 403:
                    raise MissingAuthError("Invalid secrets")
                auth_result.cookies = merge_cookies(auth_result.cookies, response)
                await raise_for_status(response)
                thinking_duration = None
                async for line in response.iter_lines():
                    if line:
                        try:
                            json_data = json.loads(line)
                            result = json_data.get("result", {})
                            if conversation_id is None:
                                conversation_id = result.get("conversation", {}).get("conversationId")
                            response_data = result.get("response", {})
                            image = response_data.get("streamingImageGenerationResponse", None)
                            if image is not None:
                                yield ImagePreview(f'{cls.assets_url}/{image["imageUrl"]}', "", {"cookies": auth_result.cookies, "headers": auth_result.headers})
                            token = response_data.get("token", result.get("token"))
                            is_thinking = response_data.get("isThinking", result.get("isThinking"))
                            if token:
                                if is_thinking:
                                    if thinking_duration is None:
                                        thinking_duration = time.time()
                                        yield Reasoning(status="🤔 Is thinking...")
                                    yield Reasoning(token)
                                else:
                                    if thinking_duration is not None:
                                        thinking_duration = time.time() - thinking_duration
                                        status = f"Thought for {thinking_duration:.2f}s" if thinking_duration > 1 else ""
                                        thinking_duration = None
                                        yield Reasoning(status=status)
                                    yield token
                            generated_images = response_data.get("modelResponse", {}).get("generatedImageUrls", None)
                            if generated_images:
                                yield ImageResponse([f'{cls.assets_url}/{image}' for image in generated_images], "", {"cookies": auth_result.cookies, "headers": auth_result.headers})
                            title = result.get("title", {}).get("newTitle", "")
                            if title:
                                yield TitleGeneration(title)
                        except json.JSONDecodeError:
                            continue
                # if conversation_id is not None:
                #     yield Conversation(conversation_id)
