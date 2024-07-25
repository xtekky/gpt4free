from __future__ import annotations
from typing import Generator, Optional, Dict, Any, Union, List
import random
import asyncio
import base64

from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..typing import AsyncResult, Messages
from ..requests import StreamSession, raise_for_status
from ..errors import ResponseError
from ..image import ImageResponse

class ReplicateHome(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://replicate.com"
    parent = "Replicate"
    working = True
    default_model = 'stability-ai/stable-diffusion-3'
    models = [
		# Models for image generation
        'stability-ai/stable-diffusion-3',
        'bytedance/sdxl-lightning-4step',
        'playgroundai/playground-v2.5-1024px-aesthetic',
        
        # Models for image generation
        'meta/meta-llama-3-70b-instruct',
        'mistralai/mixtral-8x7b-instruct-v0.1',
        'google-deepmind/gemma-2b-it',
    ]

    versions = {
		# Model versions for generating images
        'stability-ai/stable-diffusion-3': [
            "527d2a6296facb8e47ba1eaf17f142c240c19a30894f437feee9b91cc29d8e4f"
        ],
        'bytedance/sdxl-lightning-4step': [
            "5f24084160c9089501c1b3545d9be3c27883ae2239b6f412990e82d4a6210f8f"
        ],
        'playgroundai/playground-v2.5-1024px-aesthetic': [
            "a45f82a1382bed5c7aeb861dac7c7d191b0fdf74d8d57c4a0e6ed7d4d0bf7d24"
        ],
        
        
        # Model versions for text generation
        'meta/meta-llama-3-70b-instruct': [
            "dp-cf04fe09351e25db628e8b6181276547"
        ],
        'mistralai/mixtral-8x7b-instruct-v0.1': [
            "dp-89e00f489d498885048e94f9809fbc76"
        ],
        'google-deepmind/gemma-2b-it': [
            "dff94eaf770e1fc211e425a50b51baa8e4cac6c39ef074681f9e39d778773626"
        ]
    }

    image_models = {"stability-ai/stable-diffusion-3", "bytedance/sdxl-lightning-4step", "playgroundai/playground-v2.5-1024px-aesthetic"}
    text_models = {"meta/meta-llama-3-70b-instruct", "mistralai/mixtral-8x7b-instruct-v0.1", "google-deepmind/gemma-2b-it"}

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        **kwargs: Any
    ) -> Generator[Union[str, ImageResponse], None, None]:
        yield await cls.create_async(messages[-1]["content"], model, **kwargs)

    @classmethod
    async def create_async(
        cls,
        prompt: str,
        model: str,
        api_key: Optional[str] = None,
        proxy: Optional[str] = None,
        timeout: int = 180,
        version: Optional[str] = None,
        extra_data: Dict[str, Any] = {},
        **kwargs: Any
    ) -> Union[str, ImageResponse]:
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
            version = random.choice(cls.versions.get(model, []))
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
                        output = result['output']
                        if model in cls.text_models:
                            return ''.join(output) if isinstance(output, list) else output
                        elif model in cls.image_models:
                            images: List[Any] = output
                            images = images[0] if len(images) == 1 else images
                            return ImageResponse(images, prompt)
                    elif result["status"] == "failed":
                        raise ResponseError(f"Prediction failed: {result}")
                    await asyncio.sleep(0.5)
