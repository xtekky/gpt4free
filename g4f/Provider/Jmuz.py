from __future__ import annotations

from ..typing import AsyncResult, Messages
from .needs_auth.OpenaiAPI import OpenaiAPI

class Jmuz(OpenaiAPI):
    label = "Jmuz"
    url = "https://discord.gg/qXfu24JmsB"
    login_url = None
    api_base = "https://jmuz.me/gpt/api/v2"
    api_key = "prod"

    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = False

    default_model = "gpt-4o"
    model_aliases = {
        "gemini": "gemini-exp",
        "deepseek-chat": "deepseek-2.5",
        "qwq-32b": "qwq-32b-preview"
    }
    
    @classmethod
    def get_models(cls):
        if not cls.models:
            cls.models = super().get_models(api_key=cls.api_key, api_base=cls.api_base)
        return cls.models

    @classmethod
    async def create_async_generator(
            cls,
            model: str,
            messages: Messages,
            stream: bool = False,
            api_key: str = None,
            api_base: str = None,
            **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        headers = {
            "Authorization": f"Bearer {cls.api_key}",
            "Content-Type": "application/json",
            "accept": "*/*",
            "cache-control": "no-cache",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
        }
        started = False
        async for chunk in super().create_async_generator(
            model=model,
            messages=messages,
            api_base=cls.api_base,
            api_key=cls.api_key,
            stream=cls.supports_stream,
            headers=headers,
            **kwargs
        ):
            if isinstance(chunk, str) and cls.url in chunk:
                continue
            if isinstance(chunk, str) and not started:
                chunk = chunk.lstrip()
            if chunk:
                started = True
                yield chunk
