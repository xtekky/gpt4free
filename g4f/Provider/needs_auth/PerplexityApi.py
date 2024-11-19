from __future__ import annotations

from .OpenaiAPI import OpenaiAPI
from ...typing import AsyncResult, Messages

class PerplexityApi(OpenaiAPI):
    label = "Perplexity API"
    url = "https://www.perplexity.ai"
    working = True
    default_model = "llama-3-sonar-large-32k-online"
    models = [
        "llama-3-sonar-small-32k-chat",
        "llama-3-sonar-small-32k-online",
        "llama-3-sonar-large-32k-chat",
        "llama-3-sonar-large-32k-online",
        "llama-3-8b-instruct",
        "llama-3-70b-instruct",
    ]

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_base: str = "https://api.perplexity.ai",
        **kwargs
    ) -> AsyncResult:
        return super().create_async_generator(
            model, messages, api_base=api_base, **kwargs
        )
