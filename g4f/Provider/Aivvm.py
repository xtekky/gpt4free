from __future__ import annotations

from aiohttp import ClientSession

from .base_provider import AsyncGeneratorProvider
from ..typing import AsyncGenerator

models = {
    "gpt-4": {
        "id": "gpt-4",
        "name": "GPT-4",
    },
    "gpt-3.5-turbo": {
        "id": "gpt-3.5-turbo",
        "name": "GPT-3.5",
    },
    "gpt-3.5-turbo-16k": {
        "id": "gpt-3.5-turbo-16k",
        "name": "GPT-3.5-16k",
    },
}

class Aivvm(AsyncGeneratorProvider):
    url                   = "https://chat.aivvm.com"
    working               = True
    supports_gpt_35_turbo = True
    supports_gpt_4        = True


    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        **kwargs
    ) -> AsyncGenerator:
        model = model if model else "gpt-3.5-turbo"
        if model not in models:
            raise ValueError(f"Model are not supported: {model}")
        headers = {
            "User-Agent"         : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            "Accept"             : "*/*",
            "Accept-language"    : "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "Origin"             : cls.url,
            "Referer"            : cls.url + "/",
            "Sec-Fetch-Dest"     : "empty",
            "Sec-Fetch-Mode"     : "cors",
            "Sec-Fetch-Site"     : "same-origin",
        }
        async with ClientSession(
            headers=headers
        ) as session:
            data = {
                "temperature": 1,
                "key": "",
                "messages": messages,
                "model": models[model],
                "prompt": "",
                **kwargs
            }
            async with session.post(cls.url + "/api/chat", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for stream in response.content.iter_any():
                    yield stream.decode()


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("temperature", "float"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"