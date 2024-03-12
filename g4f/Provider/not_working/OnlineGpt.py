from __future__ import annotations

import json
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider
from ..helper import get_random_string

class OnlineGpt(AsyncGeneratorProvider):
    url = "https://onlinegpt.org"
    working = False
    supports_gpt_35_turbo = True
    supports_message_history = False

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
            "Accept": "text/event-stream",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{cls.url}/chat/",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Alt-Used": "onlinegpt.org",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "TE": "trailers"
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "botId": "default",
                "customId": None,
                "session": get_random_string(12),
                "chatId": get_random_string(),
                "contextId": 9,
                "messages": messages,
                "newMessage": messages[-1]["content"],
                "newImageId": None,
                "stream": True
            }
            async with session.post(f"{cls.url}/chatgpt/wp-json/mwai-ui/v1/chats/submit", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk.startswith(b"data: "):
                        data = json.loads(chunk[6:])
                        if data["type"] == "live":
                            yield data["data"]