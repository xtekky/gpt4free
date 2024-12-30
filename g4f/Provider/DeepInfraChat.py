from __future__ import annotations

import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

class DeepInfraChat(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://deepinfra.com/chat"
    api_endpoint = "https://api.deepinfra.com/v1/openai/chat/completions"
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
    models = [
        'meta-llama/Llama-3.3-70B-Instruct',
        'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'meta-llama/Llama-3.3-70B-Instruct-Turbo',
        default_model,
        'Qwen/QwQ-32B-Preview',
        'microsoft/WizardLM-2-8x22B',
        'Qwen/Qwen2.5-72B-Instruct',
        'Qwen/Qwen2.5-Coder-32B-Instruct',
        'nvidia/Llama-3.1-Nemotron-70B-Instruct',
    ]
    model_aliases = {
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "qwq-32b": "Qwen/QwQ-32B-Preview",
        "wizardlm-2-8x22b": "microsoft/WizardLM-2-8x22B",
        "qwen-2-72b": "Qwen/Qwen2.5-72B-Instruct",
        "qwen-2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
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
        model = cls.get_model(model)

        headers = {
            'Accept-Language': 'en-US,en;q=0.9',
            'Content-Type': 'application/json',
            'Origin': 'https://deepinfra.com',
            'Referer': 'https://deepinfra.com/',
            'X-Deepinfra-Source': 'web-page',
            'accept': 'text/event-stream',
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "model": model,
                "messages": messages,
                "stream": True
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                async for chunk in response.content:
                    if chunk:
                        chunk_text = chunk.decode(errors="ignore")
                        try:
                            # Handle streaming response
                            if chunk_text.startswith("data: "):
                                if chunk_text.strip() == "data: [DONE]":
                                    continue
                                chunk_data = json.loads(chunk_text[6:])
                                content = chunk_data["choices"][0]["delta"].get("content")
                                if content:
                                    yield content
                            # Handle non-streaming response
                            else:
                                chunk_data = json.loads(chunk_text)
                                content = chunk_data["choices"][0]["message"].get("content")
                                if content:
                                    yield content
                        except (json.JSONDecodeError, KeyError):
                            continue
