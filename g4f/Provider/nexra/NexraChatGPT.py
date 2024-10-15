from __future__ import annotations

from aiohttp import ClientSession
import json

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt


class NexraChatGPT(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra ChatGPT"
    url = "https://nexra.aryahcr.cc/documentation/chatgpt/en"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/gpt"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    supports_stream = False
    
    default_model = 'gpt-3.5-turbo'
    models = ['gpt-4', 'gpt-4-0613', 'gpt-4-0314', 'gpt-4-32k-0314', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-0301', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002', 'gpt-3', 'text-curie-001', 'text-babbage-001', 'text-ada-001', 'davinci', 'curie', 'babbage', 'ada', 'babbage-002', 'davinci-002']
    
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
            "Content-Type": "application/json"
        }
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "messages": messages,
                "prompt": prompt,
                "model": model,
                "markdown": False
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                response_text = await response.text()
                try:
                    if response_text.startswith('_'):
                        response_text = response_text[1:]
                    response_data = json.loads(response_text)
                    yield response_data.get('gpt', '')
                except json.JSONDecodeError:
                    yield ''
