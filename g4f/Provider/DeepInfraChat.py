from __future__ import annotations

from aiohttp import ClientSession, ClientResponseError
import json
from ..typing import AsyncResult, Messages, ImageType
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
        'Qwen/QwQ-32B-Preview',
        'microsoft/WizardLM-2-8x22B',
        'Qwen/Qwen2.5-72B-Instruct',
        'Qwen/Qwen2.5-Coder-32B-Instruct',
        'nvidia/Llama-3.1-Nemotron-70B-Instruct',
    ]
    model_aliases = {
        "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "qwq-32b": "Qwen/QwQ-32B-Preview",
        "wizardlm-2-8x22b": "microsoft/WizardLM-2-8x22B",
        "qwen-2-72b": "Qwen/Qwen2.5-72B-Instruct",
        "qwen-2.5-coder-32b": "Qwen2.5-Coder-32B-Instruct",
        "nemotron-70b": "nvidia/Llama-3.1-Nemotron-70B-Instruct",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            'Content-Type': 'application/json',
            'Origin': 'https://deepinfra.com',
            'Referer': 'https://deepinfra.com/',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'X-Deepinfra-Source': 'web-page',
            'accept': 'text/event-stream',
        }
        
        data = {
            'model': model,
            'messages': messages,
            'stream': True
        }

        async with ClientSession(headers=headers) as session:
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
