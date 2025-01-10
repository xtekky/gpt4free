from __future__ import annotations

from ..typing import AsyncResult, Messages
from .needs_auth.OpenaiAPI import OpenaiAPI

class Jmuz(OpenaiAPI):
    label = "Jmuz"
    login_url = None
    api_base = "https://jmuz.me/gpt/api/v2"
    api_key = "prod"

    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = False

    default_model = 'gpt-4o'

    @classmethod
    def get_models(cls):
        if not cls.models:
            cls.models = super().get_models(api_key=cls.api_key, api_base=cls.api_base)
        return cls.models

    @classmethod
    def get_model(cls, model: str, **kwargs) -> str:
        if model in cls.get_models():
            return model
        return cls.default_model

    @classmethod
    def create_async_generator(
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
        return super().create_async_generator(
            model=model,
            messages=messages,
            api_base=cls.api_base,
            api_key=cls.api_key,
            stream=cls.supports_stream,
            headers=headers,
            **kwargs
        )