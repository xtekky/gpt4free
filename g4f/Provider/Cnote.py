from __future__ import annotations

import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from .helper import format_prompt


class Cnote(AsyncGeneratorProvider):
    url = "https://f1.cnote.top"
    api_url = "https://p1api.xjai.pro/freeapi/chat-process"
    working = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "DNT": "1",
            "Sec-GPC": "1",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "cross-site",
            "TE": "trailers",
        }
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            system_message: str = "",
            data = {
                "prompt": prompt,
                "systemMessage": system_message,
                "temperature": 0.8,
                "top_p": 1,
            }
            async with session.post(cls.api_url, json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk:
                        try:
                            data = json.loads(chunk.decode().split("&KFw6loC9Qvy&")[-1])
                            text = data.get("text", "")
                            yield text
                        except (json.JSONDecodeError, IndexError):
                            pass
