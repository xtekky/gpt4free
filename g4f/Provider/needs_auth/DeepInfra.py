from __future__ import annotations

import requests
from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...requests import StreamSession, raise_for_status
from ...image import ImageResponse

class DeepInfra(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://deepinfra.com"
    working = True
    needs_auth = True
    supports_stream = True
    supports_message_history = True
    default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    default_image_model = ''
    image_models = []
    models = []
    
    @classmethod
    def get_models(cls, **kwargs):
        if not cls.models:
            url = 'https://api.deepinfra.com/models/featured'
            models = requests.get(url).json()
            cls.text_models = [model['model_name'] for model in models if model["type"] == "text-generation"]
            cls.image_models = [model['model_name'] for model in models if model["reported_type"] == "text-to-image"]
            cls.models = cls.text_models + cls.image_models
        return cls.models

    @classmethod
    async def create_text_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        temperature: float = 0.7,
        max_tokens: int = 1028,
        api_base: str = "https://api.deepinfra.com/v1/openai",
        **kwargs
    ) -> AsyncResult:
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
        
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        async with StreamSession(headers=headers) as session:
            async with session.post(f"{api_base}/chat/completions", json=data) as response:
                await raise_for_status(response)
                yield response

    @classmethod
    async def create_image(
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
            data = {"input": data} if model == cls.default_image_model else data
            
            async with session.post(f"{api_base.rstrip('/')}/{model}", json=data) as response:
                await raise_for_status(response)
                data = await response.json()
                images = data.get("output", data.get("images", data.get("image_url")))
                if not images:
                    raise RuntimeError(f"Response: {data}")
                images = images[0] if len(images) == 1 else images
                return ImageResponse(images, prompt)

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        prompt: str = None,
        **kwargs
    ) -> AsyncResult:
        if model in cls.image_models:
            prompt = messages[-1]["content"] if prompt is None else prompt
            yield await cls.create_image(prompt=prompt, model=model, **kwargs)
        else:
            async for response in cls.create_text_completion(
                model=model,
                messages=messages,
                stream=stream,
                **kwargs
            ):
                yield response
