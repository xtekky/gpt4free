from __future__ import annotations

from ...typing import AsyncResult, Messages
from ...requests import StreamSession, raise_for_status
from ...providers.response import ImageResponse
from ...config import DEFAULT_MODEL
from ..template import OpenaiTemplate
from ..DeepInfraChat import DeepInfraChat
from ..helper import format_media_prompt

class DeepInfra(OpenaiTemplate):
    url = "https://deepinfra.com"
    login_url = "https://deepinfra.com/dash/api_keys"
    api_base = "https://api.deepinfra.com/v1/openai"
    working = True
    active_by_default = True
    default_model = DEFAULT_MODEL
    vision_models = DeepInfraChat.vision_models
    model_aliases = DeepInfraChat.model_aliases

    @classmethod
    def get_models(cls, **kwargs):
        if not cls.models:
            cls.models = DeepInfraChat.get_models()
            cls.image_models = DeepInfraChat.image_models
        return cls.models

    @classmethod
    def get_image_models(cls, **kwargs):
        if not cls.image_models:
            cls.get_models()
        return cls.image_models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1028,
        **kwargs
    ) -> AsyncResult:
        if model in cls.get_image_models():
            yield cls.create_async_image(
                format_media_prompt(messages, prompt),
                model,
                **kwargs
            )
            return

        headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US',
            'Origin': 'https://deepinfra.com',
            'Referer': 'https://deepinfra.com/',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'X-Deepinfra-Source': 'web-embed',
        }
        async for chunk in super().create_async_generator(
            model, messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            headers=headers,
            **kwargs
        ):
            yield chunk

    @classmethod
    async def create_async_image(
        cls,
        prompt: str,
        model: str,
        api_key: str = None,
        api_base: str = "https://api.deepinfra.com/v1/inference",
        proxy: str = None,
        timeout: int = 180,
        extra_body: dict = {},
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
            data = {"prompt": prompt, **extra_body}
            data = {"input": data} if model == cls.default_model else data
            async with session.post(f"{api_base.rstrip('/')}/{model}", json=data) as response:
                await raise_for_status(response)
                data = await response.json()
                images = data.get("output", data.get("images", data.get("image_url")))
                if not images:
                    raise RuntimeError(f"Response: {data}")
                images = images[0] if len(images) == 1 else images
                return ImageResponse(images, prompt)