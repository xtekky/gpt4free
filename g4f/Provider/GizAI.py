from __future__ import annotations

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class GizAI(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://app.giz.ai/assistant"
    api_endpoint = "https://app.giz.ai/api/data/users/inferenceServer.infer"
    
    working = True
    supports_stream = False
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'chat-gemini-flash'
    models = [default_model]
    model_aliases = {"gemini-1.5-flash": "chat-gemini-flash",}

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
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'DNT': '1',
            'Origin': 'https://app.giz.ai',
            'Pragma': 'no-cache',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Not?A_Brand";v="99", "Chromium";v="130"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"'
        }
        
        prompt = format_prompt(messages)
        
        async with ClientSession(headers=headers) as session:
            data = {
                "model": model,
                "input": {
                    "messages": [{"type": "human", "content": prompt}],
                    "mode": "plan"
                },
                "noStream": True
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                if response.status == 201:
                    result = await response.json()
                    yield result['output'].strip()
                else:
                    raise Exception(f"Unexpected response status: {response.status}")
