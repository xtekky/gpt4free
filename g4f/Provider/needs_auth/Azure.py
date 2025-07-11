from __future__ import annotations

import os

from ...typing import Messages, AsyncResult
from ..template import OpenaiTemplate

class Azure(OpenaiTemplate):
    working = True
    needs_auth = True
    login_url = "https://ai.azure.com"

    @classmethod
    def get_models(cls, **kwargs) -> list[str]:
        cls.default_model = os.environ.get("AZURE_DEFAULT_MODEL", cls.default_model)
        return [cls.default_model]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        api_endpoint: str = None,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = os.environ.get("AZURE_DEFAULT_MODEL", cls.default_model)
        if not api_key:
            raise ValueError("API key is required for Azure provider")
        if not api_endpoint:
            api_endpoint = os.environ.get("AZURE_API_ENDPOINT")
        if not api_endpoint:
            raise ValueError("API endpoint is required for Azure provider")
        async for chunk in super().create_async_generator(
            model=model,
            messages=messages,
            api_key=api_key,
            api_endpoint=api_endpoint,
            **kwargs
        ):
            yield chunk