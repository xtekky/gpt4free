from __future__ import annotations

from aiohttp import ClientSession
import json

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt


class NexraQwen(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra Qwen"
    url = "https://nexra.aryahcr.cc/documentation/qwen/en"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/complements"
    working = True
    supports_stream = True
    
    default_model = 'qwen'
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
            "Content-Type": "application/json",
            "accept": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/chat",
        }
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "markdown": markdown,
                "stream": stream,
                "model": model
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                
                complete_message = ""
                
                # If streaming, process each chunk separately
                if stream:
                    async for chunk in response.content.iter_any():
                        if chunk:
                            try:
                                # Decode the chunk and split by the delimiter
                                parts = chunk.decode('utf-8').split('\x1e')
                                for part in parts:
                                    if part.strip():  # Ensure the part is not empty
                                        response_data = json.loads(part)
                                        message_part = response_data.get('message')
                                        if message_part:
                                            complete_message = message_part
                            except json.JSONDecodeError:
                                continue

                    # Yield the final complete message
                    if complete_message:
                        yield complete_message
                else:
                    # Handle non-streaming response
                    text_response = await response.text()
                    response_data = json.loads(text_response)
                    message = response_data.get('message')
                    if message:
                        yield message
