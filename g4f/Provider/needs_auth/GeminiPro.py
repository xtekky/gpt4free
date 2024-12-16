from __future__ import annotations

import base64
import json
import requests
from aiohttp import ClientSession, BaseConnector

from ...typing import AsyncResult, Messages, ImagesType
from ...image import to_bytes, is_accepted_format
from ...errors import MissingAuthError
from ...requests.raise_for_status import raise_for_status
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import get_connector
from ... import debug

class GeminiPro(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Google Gemini API"
    url = "https://ai.google.dev"
    api_base = "https://generativelanguage.googleapis.com/v1beta"

    working = True
    supports_message_history = True
    needs_auth = True

    default_model = "gemini-1.5-pro"
    default_vision_model = default_model
    fallback_models = [default_model, "gemini-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
    model_aliases = {
        "gemini-flash": "gemini-1.5-flash",
        "gemini-flash": "gemini-1.5-flash-8b",
    }

    @classmethod
    def get_models(cls, api_key: str = None, api_base: str = api_base) -> list[str]:
        if not cls.models:
            try:
                response = requests.get(f"{api_base}/models?key={api_key}")
                raise_for_status(response)
                data = response.json()
                cls.models = [
                    model.get("name").split("/").pop()
                    for model in data.get("models")
                    if "generateContent" in model.get("supportedGenerationMethods")
                ]
                cls.models.sort()
            except Exception as e:
                debug.log(e)
                cls.models = cls.fallback_models
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
        images: ImagesType = None,
        connector: BaseConnector = None,
        **kwargs
    ) -> AsyncResult:
        if not api_key:
            raise MissingAuthError('Add a "api_key"')

        model = cls.get_model(model, api_key=api_key, api_base=api_base)

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
                    "parts": [{"text": message["content"]}]
                }
                for message in messages
                if message["role"] != "system"
            ]
            if images is not None:
                for image, _ in images:
                    image = to_bytes(image)
                    contents[-1]["parts"].append({
                        "inline_data": {
                            "mime_type": is_accepted_format(image),
                            "data": base64.b64encode(image).decode()
                        }
                    })
            data = {
                "contents": contents,
                "generationConfig": {
                    "stopSequences": kwargs.get("stop"),
                    "temperature": kwargs.get("temperature"),
                    "maxOutputTokens": kwargs.get("max_tokens"),
                    "topP": kwargs.get("top_p"),
                    "topK": kwargs.get("top_k"),
                }
            }
            system_prompt = "\n".join(
                message["content"]
                for message in messages
                if message["role"] == "system"
            )
            if system_prompt:
                data["system_instruction"] = {"parts": {"text": system_prompt}}
            async with session.post(url, params=params, json=data) as response:
                if not response.ok:
                    data = await response.json()
                    data = data[0] if isinstance(data, list) else data
                    raise RuntimeError(f"Response {response.status}: {data['error']['message']}")
                if stream:
                    lines = []
                    async for chunk in response.content:
                        if chunk == b"[{\n":
                            lines = [b"{\n"]
                        elif chunk == b",\r\n" or chunk == b"]":
                            try:
                                data = b"".join(lines)
                                data = json.loads(data)
                                yield data["candidates"][0]["content"]["parts"][0]["text"]
                            except:
                                data = data.decode(errors="ignore") if isinstance(data, bytes) else data
                                raise RuntimeError(f"Read chunk failed: {data}")
                            lines = []
                        else:
                            lines.append(chunk)
                else:
                    data = await response.json()
                    candidate = data["candidates"][0]
                    if candidate["finishReason"] == "STOP":
                        yield candidate["content"]["parts"][0]["text"]
                    else:
                        yield candidate["finishReason"] + ' ' + candidate["safetyRatings"]
