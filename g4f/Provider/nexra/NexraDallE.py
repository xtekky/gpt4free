from __future__ import annotations

from aiohttp import ClientSession
import json

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...image import ImageResponse


class NexraDallE(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra DALL-E"
    url = "https://nexra.aryahcr.cc/documentation/dall-e/en"
    api_endpoint = "https://nexra.aryahcr.cc/api/image/complements"
    working = True

    default_model = 'dalle'
    models = [default_model]

    @classmethod
    def get_model(cls, model: str) -> str:
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
        # Retrieve the correct model to use
        model = cls.get_model(model)

        # Format the prompt from the messages
        prompt = messages[0]['content']

        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "model": model,
            "response": response
        }

        async with ClientSession(headers=headers) as session:
            async with session.post(cls.api_endpoint, json=payload, proxy=proxy) as response:
                response.raise_for_status()
                text_data = await response.text()

                try:
                    # Parse the JSON response
                    json_start = text_data.find('{')
                    json_data = text_data[json_start:]
                    data = json.loads(json_data)
                    
                    # Check if the response contains images
                    if 'images' in data and len(data['images']) > 0:
                        image_url = data['images'][0]
                        yield ImageResponse(image_url, prompt)
                    else:
                        yield ImageResponse("No images found in the response.", prompt)
                except json.JSONDecodeError:
                    yield ImageResponse("Failed to parse JSON. Response might not be in JSON format.", prompt)
