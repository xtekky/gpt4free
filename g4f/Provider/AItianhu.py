from __future__ import annotations

import json
from curl_cffi.requests import AsyncSession

from .base_provider import AsyncProvider, format_prompt


class AItianhu(AsyncProvider):
    url = "https://www.aitianhu.com"
    working = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        **kwargs
    ) -> str:
        data = {
            "prompt": format_prompt(messages),
            "options": {},
            "systemMessage": "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully.",
            "temperature": 0.8,
            "top_p": 1,
            **kwargs
        }
        async with AsyncSession(proxies={"https": proxy}, impersonate="chrome107", verify=False) as session:
            response = await session.post(cls.url + "/api/chat-process", json=data)
            response.raise_for_status()
            line = response.text.splitlines()[-1]
            line = json.loads(line)
            return line["text"]


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
