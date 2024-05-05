from __future__ import annotations

import requests

from .needs_auth.Openai import Openai
from ..typing import AsyncResult, Messages

class Ollama(Openai):
    label = "Ollama"
    url = "https://ollama.com"
    needs_auth = False
    working = True

    @classmethod
    def get_models(cls):
        if not cls.models:
            url = 'http://127.0.0.1:11434/api/tags'
            models = requests.get(url).json()["models"]
            cls.models = [model['name'] for model in models]
            cls.default_model = cls.models[0]
        return cls.models

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_base: str = "http://localhost:11434/v1",
        **kwargs
    ) -> AsyncResult:
        return super().create_async_generator(
            model, messages, api_base=api_base, **kwargs
        )