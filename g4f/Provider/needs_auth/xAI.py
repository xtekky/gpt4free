from __future__ import annotations

from .OpenaiAPI import OpenaiAPI
from ...typing import AsyncResult, Messages

class xAI(OpenaiAPI):
    label = "xAI"
    url = "https://console.x.ai"
    api_base = "https://api.x.ai/v1"
    working = True

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_base: str = api_base,
        **kwargs
    ) -> AsyncResult:
        return super().create_async_generator(
            model, messages, api_base=api_base, **kwargs
        )