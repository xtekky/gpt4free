from __future__ import annotations

import base64
import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages, ImageType
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import to_bytes, is_accepted_format


class GeminiPro(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://ai.google.dev"
    working = True
    supports_message_history = True
    default_model = "gemini-pro"
    models = ["gemini-pro", "gemini-pro-vision"]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        proxy: str = None,
        api_key: str = None,
        image: ImageType = None,
        **kwargs
    ) -> AsyncResult:
        model = "gemini-pro-vision" if not model and image else model
        model = cls.get_model(model)
        api_key = api_key if api_key else kwargs.get("access_token")
        headers = {
            "Content-Type": "application/json",
        }
        async with ClientSession(headers=headers) as session:
            method = "streamGenerateContent" if stream else "generateContent"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}"
            contents = [
                {
                    "role": "model" if message["role"] == "assistant" else message["role"],
                    "parts": [{"text": message["content"]}]
                }
                for message in messages
            ]
            if image:
                image = to_bytes(image)
                contents[-1]["parts"].append({
                    "inline_data": {
                        "mime_type": is_accepted_format(image),
                        "data": base64.b64encode(image).decode()
                    }
                })
            data = {
                "contents": contents,
                # "generationConfig": {
                #     "stopSequences": kwargs.get("stop"),
                #     "temperature": kwargs.get("temperature"),
                #     "maxOutputTokens": kwargs.get("max_tokens"),
                #     "topP": kwargs.get("top_p"),
                #     "topK": kwargs.get("top_k"),
                # }
            }
            async with session.post(url, params={"key": api_key}, json=data, proxy=proxy) as response:
                if not response.ok:
                    data = await response.json()
                    raise RuntimeError(data[0]["error"]["message"])
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
                                data = data.decode() if isinstance(data, bytes) else data
                                raise RuntimeError(f"Read text failed. data: {data}")
                            lines = []
                        else:
                            lines.append(chunk)
                else:
                    data = await response.json()
                    yield data["candidates"][0]["content"]["parts"][0]["text"]