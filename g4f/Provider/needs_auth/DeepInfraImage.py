from __future__ import annotations

import requests

from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...typing import AsyncResult, Messages
from ...requests import StreamSession, raise_for_status
from ...image import ImageResponse

class DeepInfraImage(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://deepinfra.com"
    parent = "DeepInfra"
    working = True
    needs_auth = True
    default_model = ''
    image_models = [default_model]

    @classmethod
    def get_models(cls):
        if not cls.models:
            url = 'https://api.deepinfra.com/models/featured'
            models = requests.get(url).json()
            cls.models = [model['model_name'] for model in models if model["reported_type"] == "text-to-image"]
            cls.image_models = cls.models
        return cls.models

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
        api_base: str = "https://api.deepinfra.com/v1/inference",
        proxy: str = None,
        timeout: int = 180,
        extra_data: dict = {},
        **kwargs
    ) -> ImageResponse:
        headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US',
            'Connection': 'keep-alive',
            'Origin': 'https://deepinfra.com',
            'Referer': 'https://deepinfra.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'X-Deepinfra-Source': 'web-embed',
            'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
        }
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        async with StreamSession(
            proxies={"all": proxy},
            headers=headers,
            timeout=timeout
        ) as session:
            model = cls.get_model(model)
            data = {"prompt": prompt, **extra_data}
            data = {"input": data} if model == cls.default_model else data
            async with session.post(f"{api_base.rstrip('/')}/{model}", json=data) as response:
                await raise_for_status(response)
                data = await response.json()
                images = data["output"] if "output" in data else data["images"]
                if not images:
                    raise RuntimeError(f"Response: {data}")
                images = images[0] if len(images) == 1 else images
                return ImageResponse(images, prompt)
