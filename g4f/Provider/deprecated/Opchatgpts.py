from __future__ import annotations

import random, string, json
from aiohttp import ClientSession

from ...typing import Messages, AsyncResult
from ..base_provider import AsyncGeneratorProvider
from ..helper import get_random_string

class Opchatgpts(AsyncGeneratorProvider):
    url = "https://opchatgpts.net"
    working = False
    supports_message_history = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None, **kwargs) -> AsyncResult:
        
        headers = {
            "User-Agent"         : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            "Accept"             : "*/*",
            "Accept-Language"    : "de,en-US;q=0.7,en;q=0.3",
            "Origin"             : cls.url,
            "Alt-Used"           : "opchatgpts.net",
            "Referer"            : f"{cls.url}/chatgpt-free-use/",
            "Sec-Fetch-Dest"     : "empty",
            "Sec-Fetch-Mode"     : "cors",
            "Sec-Fetch-Site"     : "same-origin",
        }
        async with ClientSession(
            headers=headers
        ) as session:
            data = {
                "botId": "default",
                "chatId": get_random_string(),
                "contextId": 28,
                "customId": None,
                "messages": messages,
                "newMessage": messages[-1]["content"],
                "session": "N/A",
                "stream": True
            }
            async with session.post(f"{cls.url}/wp-json/mwai-ui/v1/chats/submit", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line.startswith(b"data: "):
                        try:
                            line = json.loads(line[6:])
                            assert "type" in line
                        except:
                            raise RuntimeError(f"Broken line: {line.decode()}")
                        if line["type"] == "live":
                            yield line["data"]
                        elif line["type"] == "end":
                            break