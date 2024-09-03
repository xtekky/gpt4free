from __future__ import annotations

import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class Nexra(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://nexra.aryahcr.cc"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/gpt"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-3.5-turbo'
    models = [
        # Working with text
        'gpt-4',
        'gpt-4-0613',
        'gpt-4-32k',
        'gpt-4-0314',
        'gpt-4-32k-0314',
        
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-16k',
        'gpt-3.5-turbo-0613',
        'gpt-3.5-turbo-16k-0613',
        'gpt-3.5-turbo-0301',
        
        'gpt-3',
        'text-davinci-003',
        'text-davinci-002',
        'code-davinci-002',
        'text-curie-001',
        'text-babbage-001',
        'text-ada-001',
        'davinci',
        'curie',
        'babbage',
        'ada',
        'babbage-002',
        'davinci-002', 
    ]
    
    model_aliases = {
        "gpt-4": "gpt-4-0613",
        "gpt-4": "gpt-4-32k",
        "gpt-4": "gpt-4-0314",
        "gpt-4": "gpt-4-32k-0314",
        
        "gpt-3.5-turbo": "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo": "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo": "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo": "gpt-3.5-turbo-0301",
        
        "gpt-3": "text-davinci-003",
        "gpt-3": "text-davinci-002",
        "gpt-3": "code-davinci-002",
        "gpt-3": "text-curie-001",
        "gpt-3": "text-babbage-001",
        "gpt-3": "text-ada-001",
        "gpt-3": "text-ada-001",
        "gpt-3": "davinci",
        "gpt-3": "curie",
        "gpt-3": "babbage",
        "gpt-3": "ada",
        "gpt-3": "babbage-002",
        "gpt-3": "davinci-002",
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
            "Content-Type": "application/json",
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "messages": messages,
                "prompt": format_prompt(messages),
                "model": model,
                "markdown": False,
                "stream": False,
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                result = await response.text()
                json_result = json.loads(result)
                yield json_result["gpt"]
