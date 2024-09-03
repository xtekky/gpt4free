from __future__ import annotations

import json
from typing import AsyncGenerator

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class Snova(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://fast.snova.ai"
    api_endpoint = "https://fast.snova.ai/api/completion"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'Meta-Llama-3.1-8B-Instruct'
    models = [
        'Meta-Llama-3.1-8B-Instruct',
        'Meta-Llama-3.1-70B-Instruct',
        'Meta-Llama-3.1-405B-Instruct',
        'Samba-CoE',
        'ignos/Mistral-T5-7B-v1',
        'v1olet/v1olet_merged_dpo_7B',
        'macadeliccc/WestLake-7B-v2-laser-truthy-dpo',
        'cookinai/DonutLM-v1',
    ]
    
    model_aliases = {
        "llama-3.1-8b": "Meta-Llama-3.1-8B-Instruct",
        "llama-3.1-70b": "Meta-Llama-3.1-70B-Instruct",
        "llama-3.1-405b": "Meta-Llama-3.1-405B-Instruct",
        
        "mistral-7b": "ignos/Mistral-T5-7B-v1",
        
        "samba-coe-v0.1": "Samba-CoE",
        "v1olet-merged-7b": "v1olet/v1olet_merged_dpo_7B",
        "westlake-7b-v2": "macadeliccc/WestLake-7B-v2-laser-truthy-dpo",
        "donutlm-v1": "cookinai/DonutLM-v1",
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
    ) -> AsyncGenerator[str, None]:
        model = cls.get_model(model)
        
        headers = {
            "accept": "text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": cls.url,
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": f"{cls.url}/",
            "sec-ch-ua": '"Chromium";v="127", "Not)A;Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "body": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": format_prompt(messages),
                            "id": "1-id",
                            "ref": "1-ref",
                            "revision": 1,
                            "draft": False,
                            "status": "done",
                            "enableRealTimeChat": False,
                            "meta": None
                        }
                    ],
                    "max_tokens": 1000,
                    "stop": ["<|eot_id|>"],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "model": model
                },
                "env_type": "tp16"
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                full_response = ""
                async for line in response.content:
                    line = line.decode().strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            json_data = json.loads(data)
                            choices = json_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                full_response += content
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"Error processing chunk: {e}")
                            print(f"Problematic data: {data}")
                            continue
                
                yield full_response.strip()
