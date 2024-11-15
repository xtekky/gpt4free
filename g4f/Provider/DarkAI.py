from __future__ import annotations

import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class DarkAI(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://darkai.foundation/chat"
    api_endpoint = "https://darkai.foundation/chat"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'llama-3-405b'
    models = [
         'gpt-4o', # Uncensored
         'gpt-3.5-turbo', # Uncensored
         'llama-3-70b', # Uncensored
         default_model,
    ]
    
    model_aliases = {
        "llama-3.1-70b": "llama-3-70b",
        "llama-3.1-405b": "llama-3-405b",
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
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
            "accept": "text/event-stream",
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
        }
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "query": prompt,
                "model": model,
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                full_text = ""
                async for chunk in response.content:
                    if chunk:
                        try:
                            chunk_str = chunk.decode().strip()
                            if chunk_str.startswith('data: '):
                                chunk_data = json.loads(chunk_str[6:])
                                if chunk_data['event'] == 'text-chunk':
                                    full_text += chunk_data['data']['text']
                                elif chunk_data['event'] == 'stream-end':
                                    if full_text:
                                        yield full_text.strip()
                                    return
                        except json.JSONDecodeError:
                            pass
                        except Exception:
                            pass
                
                if full_text:
                    yield full_text.strip()
