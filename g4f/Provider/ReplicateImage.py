from __future__ import annotations

import random
import asyncio

from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..typing import AsyncResult, Messages
from ..requests import StreamSession, raise_for_status
from ..image import ImageResponse
from ..errors import ResponseError

class ReplicateImage(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://replicate.com"
    parent = "Replicate"
    working = True
    default_model = 'stability-ai/sdxl'
    default_versions = [
        "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        "2b017d9b67edd2ee1401238df49d75da53c523f36e363881e057f5dc3ed3c5b2"
    ]
    image_models = [default_model]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        **kwargs
    ) -> AsyncResult:
        yield await cls.create_async(messages[-1]["content"], model, **kwargs)

    @classmethod
    async def create_async(
        cls,
        prompt: str,
        model: str,
        api_key: str = None,
        proxy: str = None,
        timeout: int = 180,
        version: str = None,
        extra_data: dict = {},
        **kwargs
    ) -> ImageResponse:
        headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US',
            'Connection': 'keep-alive',
            'Origin': cls.url,
            'Referer': f'{cls.url}/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
        }
        if version is None:
            version = random.choice(cls.default_versions)
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        async with StreamSession(
            proxies={"all": proxy},
            headers=headers,
            timeout=timeout
        ) as session:
            data = {
                "input": {
                    "prompt": prompt,
                    **extra_data
                },
                "version": version
            }
            if api_key is None:
                data["model"] = cls.get_model(model)
                url = "https://homepage.replicate.com/api/prediction"
            else:
                url = "https://api.replicate.com/v1/predictions"
            async with session.post(url, json=data) as response:
                await raise_for_status(response)
                result = await response.json()
            if "id" not in result:
                raise ResponseError(f"Invalid response: {result}")
            while True:
                if api_key is None:
                    url = f"https://homepage.replicate.com/api/poll?id={result['id']}"
                else:
                    url = f"https://api.replicate.com/v1/predictions/{result['id']}"
                async with session.get(url) as response:
                    await raise_for_status(response)
                    result = await response.json()
                    if "status" not in result:
                        raise ResponseError(f"Invalid response: {result}")
                    if result["status"] == "succeeded":
                        images = result['output']
                        images = images[0] if len(images) == 1 else images
                        return ImageResponse(images, prompt)
                    await asyncio.sleep(0.5)