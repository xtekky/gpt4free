from __future__ import annotations

import base64
import json
from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages, ImageType
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import to_bytes, is_accepted_format
from ..errors import MissingAuthError
from .helper import get_connector

class GeminiPro(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Gemini API"
    url = "https://ai.google.dev"
    working = True
    supports_message_history = True
    needs_auth = True
    default_model = "gemini-1.5-pro-latest"
    default_vision_model = default_model
    models = [default_model, "gemini-pro", "gemini-pro-vision", "gemini-1.5-flash"]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        proxy: str = None,
        api_key: str = None,
        api_base: str = "https://generativelanguage.googleapis.com/v1beta",
        use_auth_header: bool = False,
        image: ImageType = None,
        connector: BaseConnector = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)

        if not api_key:
            raise MissingAuthError('Add a "api_key"')

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
            ]
            if image is not None:
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
                    yield data["candidates"][0]["content"]["parts"][0]["text"]