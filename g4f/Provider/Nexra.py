from __future__ import annotations

import json
import base64
from aiohttp import ClientSession
from typing import AsyncGenerator

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import ImageResponse
from .helper import format_prompt

class Nexra(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://nexra.aryahcr.cc"
    api_endpoint_text = "https://nexra.aryahcr.cc/api/chat/gpt"
    api_endpoint_image = "https://nexra.aryahcr.cc/api/image/complements"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-3.5-turbo'
    models = [
        # Text models
        'gpt-4', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-0314', 'gpt-4-32k-0314',
        'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-0301',
        'gpt-3', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002',
        'text-curie-001', 'text-babbage-001', 'text-ada-001',
        'davinci', 'curie', 'babbage', 'ada', 'babbage-002', 'davinci-002',
        # Image models
        'dalle', 'dalle-mini', 'emi'
    ]
    
    image_models = {"dalle", "dalle-mini", "emi"}
    text_models = set(models) - image_models
    
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
    ) -> AsyncGenerator[str | ImageResponse, None]:
        model = cls.get_model(model)
        
        if model in cls.image_models:
            async for result in cls.create_image_async_generator(model, messages, proxy, **kwargs):
                yield result
        else:
            async for result in cls.create_text_async_generator(model, messages, proxy, **kwargs):
                yield result

    @classmethod
    async def create_text_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
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
            async with session.post(cls.api_endpoint_text, json=data, proxy=proxy) as response:
                response.raise_for_status()
                result = await response.text()
                json_result = json.loads(result)
                yield json_result["gpt"]

    @classmethod
    async def create_image_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncGenerator[ImageResponse | str, None]:
        headers = {
            "Content-Type": "application/json"
        }

        prompt = messages[-1]['content'] if messages else ""

        data = {
            "prompt": prompt,
            "model": model
        }

        async def process_response(response_text: str) -> ImageResponse | None:
            json_start = response_text.find('{')
            if json_start != -1:
                json_data = response_text[json_start:]
                try:
                    response_data = json.loads(json_data)
                    image_data = response_data.get('images', [])[0]
                    
                    if image_data.startswith('data:image/'):
                        return ImageResponse([image_data], "Generated image")
                    
                    try:
                        base64.b64decode(image_data)
                        data_uri = f"data:image/jpeg;base64,{image_data}"
                        return ImageResponse([data_uri], "Generated image")
                    except:
                        print("Invalid base64 data")
                        return None
                except json.JSONDecodeError:
                    print("Failed to parse JSON.")
            else:
                print("No JSON data found in the response.")
            return None

        async with ClientSession(headers=headers) as session:
            async with session.post(cls.api_endpoint_image, json=data, proxy=proxy) as response:
                response.raise_for_status()
                response_text = await response.text()
                
                image_response = await process_response(response_text)
                if image_response:
                    yield image_response
                else:
                    yield "Failed to process image data."

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> str:
        async for response in cls.create_async_generator(model, messages, proxy, **kwargs):
            if isinstance(response, ImageResponse):
                return response.images[0]
            return response
