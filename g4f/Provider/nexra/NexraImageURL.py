from __future__ import annotations
from aiohttp import ClientSession
import json
from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt
from ...image import ImageResponse

class NexraImageURL(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Image Generation Provider"
    api_endpoint = "https://nexra.aryahcr.cc/api/image/complements"
    models = ['dalle', 'dalle2', 'dalle-mini', 'emi', 'sdxl-turbo', 'prodia']

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "Content-Type": "application/json",
        }
        
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "prompt": prompt,
                "model": model,
                "response": "url"
            }
            
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                response_text = await response.text()
                
                cleaned_response = response_text.lstrip('_')
                response_json = json.loads(cleaned_response)
                
                images = response_json.get("images")
                if images and len(images) > 0:
                    image_response = ImageResponse(images[0], alt="Generated Image")
                    yield image_response
                else:
                    yield "No image URL found."
