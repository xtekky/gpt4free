from __future__ import annotations

from aiohttp import ClientSession, ContentTypeError
import json

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt


class NexraChatGptWeb(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra ChatGPT Web"
    url = "https://nexra.aryahcr.cc/documentation/chatgpt/en"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/{}"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    supports_stream = True
    
    default_model = 'gptweb'
    models = [default_model]
    
    model_aliases = {
        "gpt-4": "gptweb",
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
        markdown: bool = False,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "Content-Type": "application/json"
        }
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "prompt": prompt,
                "markdown": markdown
            }
            model = cls.get_model(model)
            endpoint = cls.api_endpoint.format(model)
            async with session.post(endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                response_text = await response.text()
                
                # Remove leading underscore if present
                if response_text.startswith('_'):
                    response_text = response_text[1:]
                
                try:
                    response_data = json.loads(response_text)
                    yield response_data.get('gpt', response_text)
                except json.JSONDecodeError:
                    yield response_text
