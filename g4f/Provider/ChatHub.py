from __future__ import annotations

import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class ChatHub(AsyncGeneratorProvider, ProviderModelMixin):
    label = "ChatHub"
    url = "https://app.chathub.gg"
    api_endpoint = "https://app.chathub.gg/api/v3/chat/completions"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'meta/llama3.1-8b'
    models = [
        'meta/llama3.1-8b',
        'mistral/mixtral-8x7b',
        'google/gemma-2',
        'perplexity/sonar-online',
    ]
    
    model_aliases = {
        "llama-3.1-8b": "meta/llama3.1-8b",
        "mixtral-8x7b": "mistral/mixtral-8x7b",
        "gemma-2": "google/gemma-2",
        "sonar-online": "perplexity/sonar-online",
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
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'origin': cls.url,
            'referer': f"{cls.url}/chat/cloud-llama3.1-8b",
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            'x-app-id': 'web'
        }
        
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "tools": []
            }
            
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data:'):
                            try:
                                data = json.loads(decoded_line[5:])
                                if data['type'] == 'text-delta':
                                    yield data['textDelta']
                                elif data['type'] == 'done':
                                    break
                            except json.JSONDecodeError:
                                continue 
