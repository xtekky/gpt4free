from __future__ import annotations

from aiohttp import ClientSession

from .OpenaiAPI import OpenaiAPI
from ...typing import AsyncResult, Messages, Cookies
from ...requests.raise_for_status import raise_for_status
from ...cookies import get_cookies

class Cerebras(OpenaiAPI):
    label = "Cerebras Inference"
    url = "https://inference.cerebras.ai/"
    api_base = "https://api.cerebras.ai/v1"
    working = True
    default_model = "llama3.1-70b"
    models = [
        "llama3.1-70b",
        "llama3.1-8b",
    ]
    model_aliases = {"llama-3.1-70b": "llama3.1-70b", "llama-3.1-8b": "llama3.1-8b"}

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_base: str = api_base,
        api_key: str = None,
        cookies: Cookies = None,
        **kwargs
    ) -> AsyncResult:
        if api_key is None and cookies is None:
            cookies = get_cookies(".cerebras.ai")
        async with ClientSession(cookies=cookies) as session:
            async with session.get("https://inference.cerebras.ai/api/auth/session") as response:
                raise_for_status(response)
                data = await response.json()
                if data:
                    api_key = data.get("user", {}).get("demoApiKey")
        async for chunk in super().create_async_generator(
            model, messages,
            api_base=api_base,
            impersonate="chrome",
            api_key=api_key,
            headers={
                "User-Agent": "ex/JS 1.5.0",
            },
            **kwargs
        ):
            yield chunk