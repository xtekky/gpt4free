from __future__ import annotations

import random
from datetime import datetime

from ..typing import AsyncResult, Messages
from ..requests import StreamSession
from .base_provider import AsyncGeneratorProvider, format_prompt


class Phind(AsyncGeneratorProvider):
    url = "https://www.phind.com"
    working = True
    supports_gpt_4 = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        **kwargs
    ) -> AsyncResult:
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        user_id = ''.join(random.choice(chars) for _ in range(24))
        data = {
            "question": format_prompt(messages),
            "webResults": [],
            "options": {
                "date": datetime.now().strftime("%d.%m.%Y"),
                "language": "en",
                "detailed": True,
                "anonUserId": user_id,
                "answerModel": "GPT-4",
                "creativeMode": False,
                "customLinks": []
            },
            "context":""
        }
        headers = {
            "Authority": cls.url,
            "Accept": "application/json, text/plain, */*",
            "Origin": cls.url,
            "Referer": f"{cls.url}/"
        }
        async with StreamSession(
            headers=headers,
            timeout=(5, timeout),
            proxies={"https": proxy},
            impersonate="chrome107"
        ) as session:
            async with session.post(f"{cls.url}/api/infer/answer", json=data) as response:
                response.raise_for_status()
                new_lines = 0
                async for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        line = line[6:]
                    if line.startswith(b"<PHIND_METADATA>"):
                        continue
                    if line:
                        if new_lines:
                            yield "".join(["\n" for _ in range(int(new_lines / 2))])
                            new_lines = 0
                        yield line.decode()
                    else:
                        new_lines += 1


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("proxy", "str"),
            ("timeout", "int"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
