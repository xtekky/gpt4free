from __future__ import annotations

from ..typing import AsyncResult, Messages
from .template import OpenaiTemplate

class Mhystical(OpenaiTemplate):
    url = "https://mhystical.cc"
    api_endpoint = "https://api.mhystical.cc/v1/completions"
    login_url = "https://mhystical.cc/dashboard"
    api_key = "mhystical"

    working = True
    supports_stream = False  # Set to False, as streaming is not specified in ChatifyAI
    supports_system_message = False

    default_model = 'gpt-4'
    models = [default_model]

    @classmethod
    def get_model(cls, model: str, **kwargs) -> str:
        cls.last_model = cls.default_model
        return cls.default_model

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        api_key: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "x-api-key": cls.api_key,
            "Content-Type": "application/json",
            "accept": "*/*",
            "cache-control": "no-cache",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
        }
        return super().create_async_generator(
            model=model,
            messages=messages,
            stream=cls.supports_stream,
            api_endpoint=cls.api_endpoint,
            headers=headers,
            **kwargs
        )
