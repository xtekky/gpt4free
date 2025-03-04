import os
import json
import random
import re
import base64
import asyncio
import time
from urllib.parse import quote_plus, unquote_plus
from pathlib import Path
from aiohttp import ClientSession, BaseConnector
from typing import Dict, Any, Optional, AsyncIterator, List

from ... import debug
from ...typing import Messages, Cookies, ImagesType, AsyncResult
from ...providers.response import JsonConversation, Reasoning, ImagePreview, ImageResponse, TitleGeneration
from ...requests.raise_for_status import raise_for_status
from ...requests.aiohttp import get_connector
from ...requests import get_nodriver
from ...errors import MissingAuthError
from ...cookies import get_cookies_dir
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, get_cookies, get_last_user_message

class Conversation(JsonConversation):
    def __init__(self,
        conversation_id: str,
        response_id: str,
        choice_id: str,
        model: str
    ) -> None:
        self.conversation_id = conversation_id
        self.response_id = response_id
        self.choice_id = choice_id
        self.model = model

class Grok(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Grok AI"
    url = "https://grok.com"
    assets_url = "https://assets.grok.com"
    conversation_url = "https://grok.com/rest/app-chat/conversations"
    
    needs_auth = True
    working = False

    default_model = "grok-3"
    models = [default_model, "grok-3-thinking", "grok-2"]

    _cookies: Cookies = None

    @classmethod
    async def _prepare_payload(cls, model: str, message: str) -> Dict[str, Any]:
        return {
            "temporary": False,
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
            "isReasoning": model.endswith("-thinking"),
        }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        cookies: Cookies = None,
        connector: BaseConnector = None,
        images: ImagesType = None,
        return_conversation: bool = False,
        conversation: Optional[Conversation] = None,
        **kwargs
    ) -> AsyncResult:
        cls._cookies = cookies or cls._cookies or get_cookies(".grok.com", False, True)
        if not cls._cookies:
            raise MissingAuthError("Missing required cookies")

        prompt = format_prompt(messages) if conversation is None else get_last_user_message(messages)
        base_connector = get_connector(connector, proxy)

        headers = {
            "accept": "*/*",
            "accept-language": "en-GB,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://grok.com",
            "priority": "u=1, i",
            "referer": "https://grok.com/",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Brave";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        }

        async with ClientSession(
            headers=headers,
            cookies=cls._cookies,
            connector=base_connector
        ) as session:
            payload = await cls._prepare_payload(model, prompt)
            response = await session.post(f"{cls.conversation_url}/new", json=payload)
            await raise_for_status(response)

            thinking_duration = None
            async for line in response.content:
                if line:
                    try:
                        json_data = json.loads(line)
                        result = json_data.get("result", {})
                        response_data = result.get("response", {})
                        image = response_data.get("streamingImageGenerationResponse", None)
                        if image is not None:
                            yield ImagePreview(f'{cls.assets_url}/{image["imageUrl"]}', "", {"cookies": cookies, "headers": headers})
                        token = response_data.get("token", "")
                        is_thinking = response_data.get("isThinking", False)
                        if token:
                            if is_thinking:
                                if thinking_duration is None:
                                    thinking_duration = time.time()
                                    yield Reasoning(status="🤔 Is thinking...")
                                yield Reasoning(token)
                            else:
                                if thinking_duration is not None:
                                    thinking_duration = time.time() - thinking_duration
                                    status = f"Thought for {thinking_duration:.2f}s" if thinking_duration > 1 else "Finished"
                                    thinking_duration = None
                                    yield Reasoning(status=status)
                                yield token
                        generated_images = response_data.get("modelResponse", {}).get("generatedImageUrls", None)
                        if generated_images:
                            yield ImageResponse([f'{cls.assets_url}/{image}' for image in generated_images], "", {"cookies": cookies, "headers": headers})
                        title = response_data.get("title", {}).get("newTitle", "")
                        if title:
                            yield TitleGeneration(title)

                    except json.JSONDecodeError:
                        continue