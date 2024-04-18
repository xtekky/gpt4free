from __future__ import annotations

import requests

from .Openai import Openai
from ...typing import AsyncResult, Messages

class OpenRouter(Openai):
    label = "OpenRouter"
    url = "https://openrouter.ai"
    working = True
    default_model = "mistralai/mistral-7b-instruct:free"

    @classmethod
    def get_models(cls):
        if not cls.models:
            url = 'https://openrouter.ai/api/v1/models'
            models = requests.get(url).json()["data"]
            cls.models = [model['id'] for model in models]
        return cls.models

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_base: str = "https://openrouter.ai/api/v1",
        **kwargs
    ) -> AsyncResult:
        return super().create_async_generator(
            model, messages, api_base=api_base, **kwargs
        )