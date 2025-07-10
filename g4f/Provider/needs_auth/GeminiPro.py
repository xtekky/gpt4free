from __future__ import annotations

import base64
import json
import requests
from typing import Optional
from aiohttp import ClientSession, BaseConnector

from ...typing import AsyncResult, Messages, MediaListType
from ...image import to_bytes, is_data_an_media
from ...errors import MissingAuthError, ModelNotFoundError
from ...requests import raise_for_status, iter_lines
from ...providers.response import Usage, FinishReason
from ...image.copy_images import save_response_media
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import get_connector, to_string, format_media_prompt, get_system_prompt
from ... import debug

class GeminiPro(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Google Gemini API"
    url = "https://ai.google.dev"
    login_url = "https://aistudio.google.com/u/0/apikey"
    api_base = "https://generativelanguage.googleapis.com/v1beta"

    working = True
    supports_message_history = True
    supports_system_message = True
    needs_auth = True

    default_model = "gemini-2.5-flash-preview-04-17"
    default_vision_model = default_model
    fallback_models = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-thinking-exp",
        "gemini-2.5-flash-preview-04-17",
        "gemma-3-1b-it",
        "gemma-3-12b-it",
        "gemma-3-27b-it",
        "gemma-3-4b-it",
        "gemma-3n-e4b-it"
    ]

    @classmethod
    def get_models(cls, api_key: str = None, api_base: str = api_base) -> list[str]:
        if not api_key:
            return cls.fallback_models
        if not cls.models:
            try:
                url = f"{cls.api_base if not api_base else api_base}/models"
                response = requests.get(url, params={"key": api_key})
                raise_for_status(response)
                data = response.json()
                cls.models = [
                    model.get("name").split("/").pop()
                    for model in data.get("models")
                    if "generateContent" in model.get("supportedGenerationMethods")
                ]
                cls.models.sort()
            except Exception as e:
                debug.error(e)
                if api_key is not None:
                    raise MissingAuthError("Invalid API key")
                return cls.fallback_models
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        proxy: str = None,
        api_key: str = None,
        api_base: str = api_base,
        use_auth_header: bool = False,
        media: MediaListType = None,
        tools: Optional[list] = None,
        connector: BaseConnector = None,
        **kwargs
    ) -> AsyncResult:
        if not api_key:
            raise MissingAuthError('Add a "api_key"')

        try:
            model = cls.get_model(model, api_key=api_key, api_base=api_base)
        except ModelNotFoundError:
            pass

        headers = params = None
        if use_auth_header:
            headers = {"Authorization": f"Bearer {api_key}"}
        else:
            params = {"key": api_key}

        method = "streamGenerateContent" if stream else "generateContent"
        url = f"{api_base.rstrip('/')}/models/{model}:{method}"
        async with ClientSession(headers=headers, connector=get_connector(connector, proxy)) as session:
            contents = [
                {
                    "role": "model" if message["role"] == "assistant" else "user",
                    "parts": [{"text": to_string(message["content"])}]
                }
                for message in messages
                if message["role"] not in ["system", "developer"]
            ]
            if media is not None:
                if not contents:
                    contents.append({"role": "user", "parts": []})
                for media_data, filename in media:
                    media_data = to_bytes(media_data)
                    contents[-1]["parts"].append({
                        "inline_data": {
                            "mime_type": is_data_an_media(media_data, filename),
                            "data": base64.b64encode(media_data).decode()
                        }
                    })
            responseModalities = {"responseModalities": ["AUDIO"]} if "tts" in model else {}
            data = {
                "contents": contents,
                "generationConfig": {
                    "stopSequences": kwargs.get("stop"),
                    "temperature": kwargs.get("temperature"),
                    "maxOutputTokens": kwargs.get("max_tokens"),
                    "topP": kwargs.get("top_p"),
                    "topK": kwargs.get("top_k"),
                    **responseModalities,
                },
                 "tools": [{
                    "function_declarations": [{
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "parameters": {
                            "type": "object",
                            "properties": {key: {
                                "type": value["type"],
                                "description": value["title"]
                            } for key, value in tool["function"]["parameters"]["properties"].items()}
                        },
                    } for tool in tools]
                }] if tools else None
            }
            system_prompt = get_system_prompt(messages)
            if system_prompt:
                data["system_instruction"] = {"parts": {"text": system_prompt}}
            async with session.post(url, params=params, json=data) as response:
                if not response.ok:
                    data = await response.json()
                    data = data[0] if isinstance(data, list) else data
                    raise RuntimeError(f"Response {response.status}: {data['error']['message']}")
                if stream:
                    lines = []
                    buffer = b""
                    async for chunk in iter_lines(response.content.iter_any()):
                        buffer += chunk
                        if chunk == b"[{":
                            lines = [b"{"]
                        elif chunk == b"," or chunk == b"]":
                            try:
                                data = json.loads(b"".join(lines))
                                content = data["candidates"][0]["content"]
                                if "parts" in content and content["parts"]:
                                    if "text" in content["parts"][0]:
                                        yield content["parts"][0]["text"]
                                    elif "inlineData" in content["parts"][0]:
                                        async for media in save_response_media(
                                            content["parts"][0]["inlineData"], format_media_prompt(messages)
                                        ):
                                            yield media
                                if "finishReason" in data["candidates"][0]:
                                    yield FinishReason(data["candidates"][0]["finishReason"].lower())
                                usage = data.get("usageMetadata")
                                if usage:
                                    yield Usage(
                                        prompt_tokens=usage.get("promptTokenCount"),
                                        completion_tokens=usage.get("candidatesTokenCount"),
                                        total_tokens=usage.get("totalTokenCount")
                                    )
                            except Exception as e:
                                raise RuntimeError(f"Read chunk failed") from e
                            lines = []
                        else:
                            lines.append(chunk)
                else:
                    data = await response.json()
                    candidate = data["candidates"][0]
                    if "content" in candidate:
                        content = candidate["content"]
                        if "parts" in content and content["parts"]:
                            for part in content["parts"]:
                                if "text" in part:
                                    yield part["text"]
                                elif "inlineData" in part:
                                    async for media in save_response_media(
                                        part["inlineData"], format_media_prompt(messages)
                                    ):
                                        yield media
                    if "finishReason" in candidate:
                        yield FinishReason(candidate["finishReason"].lower())
