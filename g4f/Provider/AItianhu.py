from __future__ import annotations

import json
from aiohttp import ClientSession, http

from ..typing import AsyncGenerator
from .base_provider import AsyncGeneratorProvider, format_prompt


class AItianhu(AsyncGeneratorProvider):
    url = "https://www.aitianhu.com"
    working = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        **kwargs
    ) -> AsyncGenerator:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/116.0",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Connection": "keep-alive",
            "Referer": cls.url + "/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }
        async with ClientSession(
            headers=headers,
            version=http.HttpVersion10
        ) as session:
            data = {
                "prompt": format_prompt(messages),
                "options": {},
                "systemMessage": "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully.",
                "temperature": 0.8,
                "top_p": 1,
                **kwargs
            }
            async with session.post(
                cls.url + "/api/chat-process",
                proxy=proxy,
                json=data,
                ssl=False,
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    line = json.loads(line.decode('utf-8'))
                    token = line["detail"]["choices"][0]["delta"].get("content")
                    if token:
                        yield token


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("proxy", "str"),
            ("temperature", "float"),
            ("top_p", "int"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
