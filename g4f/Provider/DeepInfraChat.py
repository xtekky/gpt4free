from __future__ import annotations

from aiohttp import ClientSession
import json

from ..typing import AsyncResult, Messages, ImageType
from ..image import to_data_uri
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin


class DeepInfraChat(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://deepinfra.com/chat"
    api_endpoint = "https://api.deepinfra.com/v1/openai/chat/completions"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
    models = [
        'meta-llama/Meta-Llama-3.1-8B-Instruct',
        default_model,
        'microsoft/WizardLM-2-8x22B',
        'Qwen/Qwen2.5-72B-Instruct',
    ]
    model_aliases = {
        "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "wizardlm-2-8x22b": "microsoft/WizardLM-2-8x22B",
        "qwen-2-72b": "Qwen/Qwen2.5-72B-Instruct",
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
        image: ImageType = None,
        image_name: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'Origin': 'https://deepinfra.com',
            'Pragma': 'no-cache',
            'Referer': 'https://deepinfra.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'X-Deepinfra-Source': 'web-embed',
            'accept': 'text/event-stream',
            'sec-ch-ua': '"Not;A=Brand";v="24", "Chromium";v="128"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
        }
        
        async with ClientSession(headers=headers) as session:
            data = {
                'model': model,
                'messages': messages,
                'stream': True
            }

            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith('data:'):
                            json_part = decoded_line[5:].strip()
                            if json_part == '[DONE]':
                                break
                            try:
                                data = json.loads(json_part)
                                choices = data.get('choices', [])
                                if choices:
                                    delta = choices[0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                print(f"JSON decode error: {json_part}")
