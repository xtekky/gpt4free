from __future__ import annotations

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from .helper import format_prompt

import random


class Aichatos(AsyncGeneratorProvider):
    url = "https://chat10.aichatos.xyz"
    api = "https://api.binjie.fun"
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
            "Origin": "https://chat10.aichatos.xyz",
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
            userId = random.randint(1000000000000, 9999999999999)
            system_message: str = "",
            data = {
                "prompt": prompt,
                "userId": "#/chat/{userId}",
                "network": True,
                "system": system_message,
                "withoutContext": False,
                "stream": True,
            }
            async with session.post(f"{cls.api}/api/generateStream", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk:
                        yield chunk.decode()
