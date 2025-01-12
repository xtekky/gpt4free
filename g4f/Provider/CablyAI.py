from __future__ import annotations

from aiohttp import ClientSession
import json

from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class CablyAI(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://cablyai.com"
    api_endpoint = "https://cablyai.com/v1/chat/completions"
    
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "Cably-80B"
    models = [default_model]
    
    model_aliases = {"cably-80b": default_model}

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:      
        model = cls.get_model(model)
        headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Content-Type': 'application/json',
            'Origin': 'https://cablyai.com',
            'Referer': 'https://cablyai.com/chat',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }
        
        async with ClientSession(headers=headers) as session:
            data = {
                "model": model,
                "messages": messages,
                "stream": stream
            }
            
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                buffer = ""
                async for chunk in response.content:
                    if chunk:
                        buffer += chunk.decode()
                        while "\n\n" in buffer:
                            chunk_data, buffer = buffer.split("\n\n", 1)
                            if chunk_data.startswith("data: "):
                                try:
                                    json_data = json.loads(chunk_data[6:])
                                    if "choices" in json_data and json_data["choices"]:
                                        content = json_data["choices"][0]["delta"].get("content", "")
                                        if content:
                                            yield content
                                except json.JSONDecodeError:
                                    # Skip invalid JSON
                                    pass
                            elif chunk_data.strip() == "data: [DONE]":
                                return
