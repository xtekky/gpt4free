from __future__ import annotations

import json
from aiohttp import ClientSession

from .base_provider import AsyncGeneratorProvider
from ..typing import AsyncGenerator

class Ylokh(AsyncGeneratorProvider):
    url                   = "https://chat.ylokh.xyz"
    working               = True
    supports_gpt_35_turbo = True


    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = True,
        proxy: str = None,
        **kwargs
    ) -> AsyncGenerator:
        model = model if model else "gpt-3.5-turbo"
        headers = {
            "User-Agent"         : "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/116.0",
            "Accept"             : "*/*",
            "Accept-language"    : "de,en-US;q=0.7,en;q=0.3",
            "Origin"             : cls.url,
            "Referer"            : cls.url + "/",
            "Sec-Fetch-Dest"     : "empty",
            "Sec-Fetch-Mode"     : "cors",
            "Sec-Fetch-Site"     : "same-origin",
        }
        data = {
            "messages": messages,
            "model": model,
            "temperature": 1,
            "presence_penalty": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "allow_fallback": True,
            "stream": stream,
            **kwargs
        }
        async with ClientSession(
            headers=headers
        ) as session:
            async with session.post("https://chatapi.ylokh.xyz/v1/chat/completions", json=data, proxy=proxy) as response:
                response.raise_for_status()
                if stream:
                    async for line in response.content:
                        line = line.decode()
                        if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                            line = json.loads(line[6:-1])
                            content = line["choices"][0]["delta"].get("content")
                            if content:
                                yield content
                else:
                    chat = await response.json()
                    yield chat["choices"][0]["message"].get("content")



    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("proxy", "str"),
            ("temperature", "float"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"