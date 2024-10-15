from __future__ import annotations

import json
from aiohttp import ClientSession
from ...image import ImageResponse

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin


class NexraSD15(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra Stable Diffusion 1.5"
    url = "https://nexra.aryahcr.cc/documentation/stable-diffusion/en"
    api_endpoint = "https://nexra.aryahcr.cc/api/image/complements"
    working = False
    
    default_model = 'stablediffusion-1.5'
    models = [default_model]
    
    model_aliases = {
        "sd-1.5": "stablediffusion-1.5",
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
        response: str = "url",  # base64 or url
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "Content-Type": "application/json",
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "prompt": messages,
                "model": model,
                "response": response
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                text_response = await response.text()

                # Clean the response by removing unexpected characters
                cleaned_response = text_response.strip('__')

                if not cleaned_response.strip():
                    raise ValueError("Received an empty response from the server.")

                try:
                    json_response = json.loads(cleaned_response)
                    image_url = json_response.get("images", [])[0]
                    # Create an ImageResponse object
                    image_response = ImageResponse(images=image_url, alt="Generated Image")
                    yield image_response
                except json.JSONDecodeError:
                    raise ValueError("Unable to decode JSON from the received text response.")
