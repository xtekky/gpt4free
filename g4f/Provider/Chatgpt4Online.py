from __future__ import annotations

import json
from aiohttp import ClientSession

from ..typing       import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider


class Chatgpt4Online(AsyncGeneratorProvider):
    url                   = "https://chatgpt4online.org"
    supports_gpt_35_turbo = True
    working               = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        async with ClientSession() as session:
            data = {
                "botId": "default",
                "customId": None,
                "session": "N/A",
                "chatId": "",
                "contextId": 58,
                "messages": messages,
                "newMessage": messages[-1]["content"],
                "stream": True
            }
            async with session.post(cls.url + "/wp-json/mwai-ui/v1/chats/submit", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line.startswith(b"data: "):
                        line = json.loads(line[6:])
                        if line["type"] == "live":
                            yield line["data"]