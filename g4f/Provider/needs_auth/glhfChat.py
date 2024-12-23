from __future__ import annotations

import requests
from aiohttp import ClientSession
import json

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt
from ...errors import ModelNotFoundError, MissingAuthError


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
    def get_model(cls, model: str, api_key: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            response = requests.get("https://glhf.chat/api/openai/v1/models", headers={
                "authorization": f"Bearer {api_key}"
            })
            if not response.ok:
                raise RuntimeError(f"Request failed to models list endpoint with status {response.status_code}")  
            data = response.json().get("data", [])
            if model in [data[i]["id"] for i in range(len(data))]:
                return model
            else:
                raise ModelNotFoundError("Cannot found this model on your account. Make sure you have loaded this model first.")

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        stream: bool = False,
        api_key: str = None,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if api_key is None:
            raise MissingAuthError("Missing API Key!")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        model = cls.get_model(model, api_key)
        
        async with ClientSession(headers=headers) as session:
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stream": stream
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                if stream:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()

                        if line == "":
                            continue

                        if line == "[DONE]":
                            break

                        if line.startswith("data: "):
                            line = line[6:]
                            try:
                                chunk = json.loads(line)
                                choices = chunk.get("choices", [])
                                if not choices:
                                    continue

                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")

                                if content:
                                    yield content

                            except (json.JSONDecodeError, IndexError) as e:
                                break
                else:
                    response_json = await response.json()
                    choices = response_json.get("choices", [])
                    for choice in choices:
                        content = choice.get("message", {}).get("content", "")
                        yield content
