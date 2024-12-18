from __future__ import annotations

from aiohttp import ClientSession
import json

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt

class glhfChat(AsyncGeneratorProvider, ProviderModelMixin):
    label = "glhf Chat"
    url = "https://glhf.chat"
    api_endpoint = "https://glhf.chat/api/openai/v1/chat/completions"
    working = True
    needs_auth = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'hf:Qwen/Qwen2.5-Coder-32B-Instruct'

    # glhf.chat supports all public models on HuggingFace that you have loaded on your account
    models = [
        'hf:Qwen/Qwen2.5-Coder-32B-Instruct',
        'hf:meta-llama/Llama-3.1-405B-Instruct',
        'hf:meta-llama/Llama-3.1-70B-Instruct',
        'hf:meta-llama/Llama-3.1-8B-Instruct',
        'hf:meta-llama/Llama-3.2-3B-Instruct',
        'hf:meta-llama/Llama-3.2-11B-Vision-Instruct',
        'hf:meta-llama/Llama-3.2-90B-Vision-Instruct',
        'hf:Qwen/Qwen2.5-72B-Instruct',
        'hf:meta-llama/Llama-3.3-70B-Instruct',
        'hf:google/gemma-2-9b-it',
        'hf:google/gemma-2-27b-it',
        'hf:mistralai/Mistral-7B-Instruct-v0.3',
        'hf:mistralai/Mixtral-8x7B-Instruct-v0.1',
        'hf:mistralai/Mixtral-8x22B-Instruct-v0.1',
        'hf:NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO',
        'hf:Qwen/Qwen2.5-7B-Instruct',
        'hf:upstage/SOLAR-10.7B-Instruct-v1.0',
        'hf:nvidia/Llama-3.1-Nemotron-70B-Instruct-HF'
    ]
    
    model_aliases = {
        'Qwen2.5-Coder-32B-Instruct': 'hf:Qwen/Qwen2.5-Coder-32B-Instruct',
        'Llama-3.1-405B-Instruct': 'hf:meta-llama/Llama-3.1-405B-Instruct',
        'Llama-3.1-70B-Instruct': 'hf:meta-llama/Llama-3.1-70B-Instruct',
        'Llama-3.1-8B-Instruct': 'hf:meta-llama/Llama-3.1-8B-Instruct',
        'Llama-3.2-3B-Instruct': 'hf:meta-llama/Llama-3.2-3B-Instruct',
        'Llama-3.2-11B-Vision-Instruct': 'hf:meta-llama/Llama-3.2-11B-Vision-Instruct',
        'Llama-3.2-90B-Vision-Instruct': 'hf:meta-llama/Llama-3.2-90B-Vision-Instruct',
        'Qwen2.5-72B-Instruct': 'hf:Qwen/Qwen2.5-72B-Instruct',
        'Llama-3.3-70B-Instruct': 'hf:meta-llama/Llama-3.3-70B-Instruct',
        'gemma-2-9b-it': 'hf:google/gemma-2-9b-it',
        'gemma-2-27b-it': 'hf:google/gemma-2-27b-it',
        'Mistral-7B-Instruct-v0.3': 'hf:mistralai/Mistral-7B-Instruct-v0.3',
        'Mixtral-8x7B-Instruct-v0.1': 'hf:mistralai/Mixtral-8x7B-Instruct-v0.1',
        'Mixtral-8x22B-Instruct-v0.1': 'hf:mistralai/Mixtral-8x22B-Instruct-v0.1',
        'Nous-Hermes-2-Mixtral-8x7B-DPO': 'hf:NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO',
        'Qwen2.5-7B-Instruct': 'hf:Qwen/Qwen2.5-7B-Instruct',
        'SOLAR-10.7B-Instruct-v1.0': 'hf:upstage/SOLAR-10.7B-Instruct-v1.0',
        'Llama-3.1-Nemotron-70B-Instruct-HF': 'hf:nvidia/Llama-3.1-Nemotron-70B-Instruct-HF'
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
        stream: bool = False,
        api_key: str = None,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        async with ClientSession(headers=headers) as session:
            data = {
                "model": model,
                "messages": messages,
                "stream": stream
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_any():
                    if chunk:
                        message = json.loads(chunk.decode())
                        yield message["choices"][0]["message"]["content"]
