from __future__ import annotations

import base64
import json
import requests
import random
from typing import Optional
from aiohttp import ClientSession, BaseConnector

from ...typing import AsyncResult, Messages, MediaListType
from ...image import to_bytes, is_data_an_media
from ...errors import MissingAuthError, ModelNotFoundError
from ...requests.raise_for_status import raise_for_status
from ...providers.response import Usage, FinishReason
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import get_connector
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

    default_model = "gemini-1.5-pro"
    default_vision_model = default_model
    fallback_models = [default_model, "gemini-2.0-flash-exp", "gemini-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
    model_aliases = {
        "gemini-1.5-pro": [default_model, "gemini-pro"],
        "gemini-1.5-flash": ["gemini-1.5-flash", "gemini-1.5-flash-8b"],
        "gemini-2.0-flash": "gemini-2.0-flash-exp",
    }

    @classmethod
    def get_models(cls, api_key: str = None, api_base: str = api_base) -> list[str]:
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
    def get_model(cls, model: str, **kwargs) -> str:
        """Get the internal model name from the user-provided model name."""
        # kwargs can contain api_key, api_base, etc. but we don't need them for model selection
        if not model:
            return cls.default_model
        
        # Check if the model exists directly in our models list
        if model in cls.models:
            return model
        
        # Check if there's an alias for this model
        if model in cls.model_aliases:
            alias = cls.model_aliases[model]
            # If the alias is a list, randomly select one of the options
            if isinstance(alias, list):
                import random
                selected_model = random.choice(alias)
                debug.log(f"GeminiPro: Selected model '{selected_model}' from alias '{model}'")
                return selected_model
            debug.log(f"GeminiPro: Using model '{alias}' for alias '{model}'")
            return alias
        
        raise ModelNotFoundError(f"Model {model} not found")

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
            if media is not None:
                for media_data, filename in media:
                    image = to_bytes(image)
                    contents[-1]["parts"].append({
                        "inline_data": {
                            "mime_type": is_data_an_media(image, filename),
                            "data": base64.b64encode(media_data).decode()
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
                                if "finishReason" in data["candidates"][0]:
                                    yield FinishReason(data["candidates"][0]["finishReason"].lower())
                                usage = data.get("usageMetadata")
                                if usage:
                                    yield Usage(
                                        prompt_tokens=usage.get("promptTokenCount"),
                                        completion_tokens=usage.get("candidatesTokenCount"),
                                        total_tokens=usage.get("totalTokenCount")
                                    )
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
