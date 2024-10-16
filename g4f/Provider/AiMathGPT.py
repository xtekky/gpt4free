from __future__ import annotations

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class AiMathGPT(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://aimathgpt.forit.ai"
    api_endpoint = "https://aimathgpt.forit.ai/api/ai"
    working = True
    supports_stream = False
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'llama3'
    models = ['llama3']
    
    model_aliases = {"llama-3.1-70b": "llama3",}

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
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'no-cache',
            'content-type': 'application/json',
            'origin': cls.url,
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': f'{cls.url}/',
            'sec-ch-ua': '"Chromium";v="129", "Not=A?Brand";v="8"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
        }

        async with ClientSession(headers=headers) as session:
            data = {
                "messages": [
                    {
                        "role": "system",
                        "content": ""
                    },
                    {
                        "role": "user",
                        "content": format_prompt(messages)
                    }
                ],
                "model": model
            }
            
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                response_data = await response.json()
                filtered_response = response_data['result']['response']
                yield filtered_response
