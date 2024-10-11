from __future__ import annotations

from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt
import json

class NexraChatGPT4o(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra ChatGPT4o"
    url = "https://nexra.aryahcr.cc/documentation/chatgpt/en"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/complements"
    working = True
    supports_gpt_4 = True
    supports_stream = False
    
    default_model = 'gpt-4o'
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
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "Content-Type": "application/json",
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": format_prompt(messages)
                    }
                ],
                "stream": False,
                "markdown": False,
                "model": model
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                buffer = ""
                last_message = ""
                async for chunk in response.content.iter_any():
                    chunk_str = chunk.decode()
                    buffer += chunk_str
                    while '{' in buffer and '}' in buffer:
                        start = buffer.index('{')
                        end = buffer.index('}', start) + 1
                        json_str = buffer[start:end]
                        buffer = buffer[end:]
                        try:
                            json_obj = json.loads(json_str)
                            if json_obj.get("finish"):
                                if last_message:
                                    yield last_message
                                return
                            elif json_obj.get("message"):
                                last_message = json_obj["message"]
                        except json.JSONDecodeError:
                            pass
                
                if last_message:
                    yield last_message
