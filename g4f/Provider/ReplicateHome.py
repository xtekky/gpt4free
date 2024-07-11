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
    default_model = 'stability-ai/sdxl'
    models = [
		# image
        'stability-ai/sdxl',
        'ai-forever/kandinsky-2.2',
        
        # text
        'meta/llama-2-70b-chat',
        'mistralai/mistral-7b-instruct-v0.2'
    ]

    versions = {
		# image
        'stability-ai/sdxl': [
            "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            "2b017d9b67edd2ee1401238df49d75da53c523f36e363881e057f5dc3ed3c5b2",
            "7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc"
        ],
        'ai-forever/kandinsky-2.2': [
            "ad9d7879fbffa2874e1d909d1d37d9bc682889cc65b31f7bb00d2362619f194a"
        ],

        
        # Text
        'meta/llama-2-70b-chat': [
            "dp-542693885b1777c98ef8c5a98f2005e7"
        ],
        'mistralai/mistral-7b-instruct-v0.2': [
            "dp-89e00f489d498885048e94f9809fbc76"
        ]
    }

    image_models = {"stability-ai/sdxl", "ai-forever/kandinsky-2.2"}
    text_models = {"meta/llama-2-70b-chat", "mistralai/mistral-7b-instruct-v0.2"}

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
