from __future__ import annotations

import requests
import os

from ..needs_auth.OpenaiAPI import OpenaiAPI
from ...typing import AsyncResult, Messages

class Ollama(OpenaiAPI):
    label = "Ollama"
    url = "https://ollama.com"
    login_url = None
    needs_auth = False
    working = True

    @classmethod
    def get_models(cls, api_base: str = None, **kwargs):
        if not cls.models:
            if api_base is None:
                host = os.getenv("OLLAMA_HOST", "127.0.0.1")
                port = os.getenv("OLLAMA_PORT", "11434")
                url = f"http://{host}:{port}/api/tags"
            else:
                url = api_base.replace("/v1", "/api/tags")
            models = requests.get(url).json()["models"]
            cls.models = [model["name"] for model in models]
            cls.default_model = cls.models[0]
        return cls.models

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_base: str = None,
        **kwargs
    ) -> AsyncResult:
        if api_base is None:
            host = os.getenv("OLLAMA_HOST", "localhost")
            port = os.getenv("OLLAMA_PORT", "11434")
            api_base: str = f"http://{host}:{port}/v1"
        return super().create_async_generator(
            model, messages, api_base=api_base, **kwargs
        )