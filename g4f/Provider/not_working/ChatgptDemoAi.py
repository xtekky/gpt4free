from __future__ import annotations

import json
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider
from ..helper import get_random_string

class ChatgptDemoAi(AsyncGeneratorProvider):
    url = "https://chat.chatgptdemo.ai"
    working = False
    supports_gpt_35_turbo = True
    supports_message_history = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Accept": "*/*",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{cls.url}/",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "TE": "trailers"
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "botId": "default",
                "customId": "8824fe9bdb323a5d585a3223aaa0cb6e",
                "session": "N/A",
                "chatId": get_random_string(12),
                "contextId": 2,
                "messages": messages,
                "newMessage": messages[-1]["content"],
                "stream": True
            }
            async with session.post(f"{cls.url}/wp-json/mwai-ui/v1/chats/submit", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    response.raise_for_status()
                    if chunk.startswith(b"data: "):
                        data = json.loads(chunk[6:])
                        if data["type"] == "live":
                            yield data["data"]