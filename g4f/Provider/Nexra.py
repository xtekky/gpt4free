from __future__ import annotations
import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..image import ImageResponse

class Nexra(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://nexra.aryahcr.cc"
    chat_api_endpoint = "https://nexra.aryahcr.cc/api/chat/gpt"
    image_api_endpoint = "https://nexra.aryahcr.cc/api/image/complements"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-3.5-turbo'
    text_models = [
        'gpt-4', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-0314', 'gpt-4-32k-0314',
        'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-0301',
        'gpt-3', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002',
        'text-curie-001', 'text-babbage-001', 'text-ada-001',
        'davinci', 'curie', 'babbage', 'ada', 'babbage-002', 'davinci-002',
    ]
    image_models = ['dalle', 'dalle2', 'dalle-mini', 'emi']
    models = [*text_models, *image_models]
    
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
        
        "dalle-2": "dalle2",
    }
    
    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.text_models or model in cls.image_models:
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
            if model in cls.image_models:
                # Image generation
                prompt = messages[-1]['content'] if messages else ""
                data = {
                    "prompt": prompt,
                    "model": model,
                    "response": "url"
                }
                async with session.post(cls.image_api_endpoint, json=data, proxy=proxy) as response:
                    response.raise_for_status()
                    result = await response.text()
                    result_json = json.loads(result.strip('_'))
                    image_url = result_json['images'][0] if result_json['images'] else None
                    
                    if image_url:
                        yield ImageResponse(images=image_url, alt=prompt)
            else:
                # Text completion
                data = {
                    "messages": messages,
                    "prompt": format_prompt(messages),
                    "model": model,
                    "markdown": False
                }
                async with session.post(cls.chat_api_endpoint, json=data, proxy=proxy) as response:
                    response.raise_for_status()
                    result = await response.text()
                    
                    try:
                        json_response = json.loads(result)
                        gpt_response = json_response.get('gpt', '')
                        yield gpt_response
                    except json.JSONDecodeError:
                        yield result
