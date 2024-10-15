from __future__ import annotations

from aiohttp import ClientSession
import json
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt
from ...typing import AsyncResult, Messages


class NexraGeminiPro(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra Gemini PRO"
    url = "https://nexra.aryahcr.cc/documentation/gemini-pro/en"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/complements"
    working = False
    supports_stream = True

    default_model = 'gemini-pro'
    models = [default_model]

    @classmethod
    def get_model(cls, model: str) -> str:
        return cls.default_model

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        stream: bool = False,
        markdown: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": format_prompt(messages)
                }
            ],
            "markdown": markdown,
            "stream": stream,
            "model": model
        }

        async with ClientSession(headers=headers) as session:
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                buffer = ""
                async for chunk in response.content.iter_any():
                    if chunk.strip():  # Check if chunk is not empty
                        buffer += chunk.decode()
                        while '\x1e' in buffer:
                            part, buffer = buffer.split('\x1e', 1)
                            if part.strip():
                                try:
                                    response_json = json.loads(part)
                                    message = response_json.get("message", "")
                                    if message:
                                        yield message
                                except json.JSONDecodeError as e:
                                    print(f"JSONDecodeError: {e}")
