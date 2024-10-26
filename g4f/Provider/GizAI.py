from __future__ import annotations

import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from ..image import ImageResponse
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class GizAI(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://app.giz.ai/assistant/"
    api_endpoint = "https://app.giz.ai/api/data/users/inferenceServer.infer"
    working = True
    
    supports_system_message = True
    supports_message_history = True
    
    # Chat models
    default_model = 'chat-gemini-flash'
    chat_models = [
        default_model,
        'chat-gemini-pro',
        'chat-gpt4m',
        'chat-gpt4',
        'claude-sonnet',
        'claude-haiku',
        'llama-3-70b',
        'llama-3-8b',
        'mistral-large',
        'chat-o1-mini'
    ]

    # Image models
    image_models = [
        'flux1',
        'sdxl',
        'sd',
        'sd35',
    ]

    models = [*chat_models, *image_models]
    
    model_aliases = {
        # Chat model aliases
        "gemini-flash": "chat-gemini-flash",
        "gemini-pro": "chat-gemini-pro",
        "gpt-4o-mini": "chat-gpt4m",
        "gpt-4o": "chat-gpt4",
        "claude-3.5-sonnet": "claude-sonnet",
        "claude-3-haiku": "claude-haiku",
        "llama-3.1-70b": "llama-3-70b",
        "llama-3.1-8b": "llama-3-8b",
        "o1-mini": "chat-o1-mini",
        # Image model aliases
        "sd-1.5": "sd",
        "sd-3.5": "sd35",
        "flux-schnell": "flux1",
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
    def is_image_model(cls, model: str) -> bool:
        return model in cls.image_models

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

        async with ClientSession() as session:
            if cls.is_image_model(model):
                # Image generation
                prompt = messages[-1]["content"]
                data = {
                    "model": model,
                    "input": {
                        "width": "1024",
                        "height": "1024",
                        "steps": 4,
                        "output_format": "webp",
                        "batch_size": 1,
                        "mode": "plan",
                        "prompt": prompt
                    }
                }
                async with session.post(
                    cls.api_endpoint,
                    headers=headers,
                    data=json.dumps(data),
                    proxy=proxy
                ) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    if response_data.get('status') == 'completed' and response_data.get('output'):
                        for url in response_data['output']:
                            yield ImageResponse(images=url, alt="Generated Image")
            else:
                # Chat completion
                data = {
                    "model": model,
                    "input": {
                        "messages": [
                            {
                                "type": "human",
                                "content": format_prompt(messages)
                            }
                        ],
                        "mode": "plan"
                    },
                    "noStream": True
                }
                async with session.post(
                    cls.api_endpoint,
                    headers=headers,
                    data=json.dumps(data),
                    proxy=proxy
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    yield result.get('output', '')
