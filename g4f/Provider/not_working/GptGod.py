from __future__ import annotations

import secrets
import json
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider
from ..helper import format_prompt

class GptGod(AsyncGeneratorProvider):
    url = "https://gptgod.site"
    working = False
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
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/118.0",
            "Accept": "text/event-stream",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Alt-Used": "gptgod.site",
            "Connection": "keep-alive",
            "Referer": f"{cls.url}/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
        }

        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "content": prompt,
                "id": secrets.token_hex(16).zfill(32)
            }
            async with session.get(f"{cls.url}/api/session/free/gpt3p5", params=data, proxy=proxy) as response:
                response.raise_for_status()
                event = None
                async for line in response.content:
                   # print(line)

                    if line.startswith(b'event: '):
                        event = line[7:-1]
                    
                    elif event == b"data" and line.startswith(b"data: "):
                        data = json.loads(line[6:-1])
                        if data:
                            yield data
                    
                    elif event == b"done":
                        break