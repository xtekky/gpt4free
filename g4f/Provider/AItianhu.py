from __future__ import annotations

import json

from ..typing import AsyncResult, Messages
from ..requests import StreamSession
from .base_provider import AsyncGeneratorProvider, format_prompt, get_cookies


class AItianhu(AsyncGeneratorProvider):
    url = "https://www.aitianhu.com"
    working = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        cookies: dict = None,
        timeout: int = 120,
        **kwargs
    ) -> AsyncResult:
        if not cookies:
            cookies = get_cookies("www.aitianhu.com")
        data = {
            "prompt": format_prompt(messages),
            "options": {},
            "systemMessage": "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully.",
            "temperature": 0.8,
            "top_p": 1,
            **kwargs
        }
        headers = {
            "Authority": cls.url,
            "Accept": "application/json, text/plain, */*",
            "Origin": cls.url,
            "Referer": f"{cls.url}/"
        }
        async with StreamSession(
            headers=headers,
            cookies=cookies,
            timeout=timeout,
            proxies={"https": proxy},
            impersonate="chrome107",
            verify=False
        ) as session:
            async with session.post(f"{cls.url}/api/chat-process", json=data) as response:
                response.raise_for_status()
                async for line in response.iter_lines():
                    if line == b"<script>":
                        raise RuntimeError("Solve challenge and pass cookies")
                    if b"platform's risk control" in line:
                        raise RuntimeError("Platform's Risk Control")
                    line = json.loads(line)
                    if "detail" in line:
                        content = line["detail"]["choices"][0]["delta"].get("content")
                        if content:
                            yield content
                    else:
                        raise RuntimeError(f"Response: {line}")


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
