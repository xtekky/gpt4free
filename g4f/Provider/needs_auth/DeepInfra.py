from __future__ import annotations

import requests
from ...typing import AsyncResult, Messages
from .OpenaiAPI import OpenaiAPI
from ...requests import StreamSession, raise_for_status
from ...image import ImageResponse

class DeepInfra(OpenaiAPI):
    label = "DeepInfra"
    url = "https://deepinfra.com"
    login_url = "https://deepinfra.com/dash/api_keys"
    working = True
    api_base = "https://api.deepinfra.com/v1/openai",
    needs_auth = True
    supports_stream = True
    supports_message_history = True
    default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    default_image_model = ''
    image_models = [default_image_model]

    @classmethod
    def get_models(cls, **kwargs):
        if not cls.models:
            url = 'https://api.deepinfra.com/models/featured'
            models = requests.get(url).json()
            cls.models = [model['model_name'] for model in models if model["type"] == "text-generation"]
            cls.image_models = [model['model_name'] for model in models if model["reported_type"] == "text-to-image"]
        return cls.models

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 1028,
        prompt: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US',
            'Origin': 'https://deepinfra.com',
            'Referer': 'https://deepinfra.com/',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'X-Deepinfra-Source': 'web-embed',
        }

        # Check if the model is an image model
        if model in cls.image_models:
            return cls.create_image_generator(messages[-1]["content"] if prompt is None else prompt, model, headers=headers, **kwargs)
        
        # Text generation
        return super().create_async_generator(
            model, messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            headers=headers,
            **kwargs
        )

    @classmethod
    async def create_image_generator(
        cls,
        prompt: str,
        model: str,
        api_key: str = None,
        api_base: str = "https://api.deepinfra.com/v1/inference",
        proxy: str = None,
        timeout: int = 180,
        headers: dict = None,
        extra_data: dict = {},
        **kwargs
    ) -> AsyncResult:
        if api_key is not None and headers is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        
        async with StreamSession(
            proxies={"all": proxy},
            headers=headers,
            timeout=timeout
        ) as session:
            model = cls.get_model(model)
            data = {"prompt": prompt, **extra_data}
            data = {"input": data} if model == cls.default_image_model else data
            async with session.post(f"{api_base.rstrip('/')}/{model}", json=data) as response:
                await raise_for_status(response)
                data = await response.json()
                images = data.get("output", data.get("images", data.get("image_url")))
                if not images:
                    raise RuntimeError(f"Response: {data}")
                images = images[0] if len(images) == 1 else images
                yield ImageResponse(images, prompt)
