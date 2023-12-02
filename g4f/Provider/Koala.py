from __future__ import annotations

import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from .helper import get_random_string

class Koala(AsyncGeneratorProvider):
    url = "https://koala.sh"
    supports_gpt_35_turbo = True
    supports_message_history = True
    working = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = "gpt-3.5-turbo"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Accept": "text/event-stream",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{cls.url}/chat",
            "Content-Type": "application/json",
            "Flag-Real-Time-Data": "false",
            "Visitor-ID": get_random_string(20),
            "Origin": cls.url,
            "Alt-Used": "koala.sh",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
            "TE": "trailers",
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "input": messages[-1]["content"],
                "inputHistory": [
                    message["content"]
                    for message in messages
                    if message["role"] == "user"
                ],
                "outputHistory": [
                    message["content"]
                    for message in messages
                    if message["role"] == "assistant"
                ],
                "model": model,
            }
            async with session.post(f"{cls.url}/api/gpt/", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk.startswith(b"data: "):
                        yield json.loads(chunk[6:])