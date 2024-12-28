from __future__ import annotations

from ..typing import AsyncResult, Messages
from .needs_auth.OpenaiAPI import OpenaiAPI

"""
    Mhystical.cc
    ~~~~~~~~~~~~
    Author: NoelP.dev
    Last Updated: 2024-05-11
    
    Author Site: https://noelp.dev
    Provider Site: https://mhystical.cc

"""

class Mhystical(OpenaiAPI):
    url = "https://api.mhystical.cc"
    api_endpoint = "https://api.mhystical.cc/v1/completions"
    working = True
    needs_auth = False
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
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        headers = {
            "x-api-key": "mhystical",
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