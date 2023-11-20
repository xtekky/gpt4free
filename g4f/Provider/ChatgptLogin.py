from __future__ import annotations

import re
import time
import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from .helper import format_prompt


class ChatgptLogin(AsyncGeneratorProvider):
    url                   = "https://chatgptlogin.ai"
    supports_gpt_35_turbo = True
    working               = False
    _user_id              = None

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
            "Accept": "*/*",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{cls.url}/chat/",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Alt-Used": "chatgptlogin.ai",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache"
        }
        async with ClientSession(headers=headers) as session:
            if not cls._user_id:
                async with session.get(f"{cls.url}/chat/", proxy=proxy) as response:
                    response.raise_for_status()
                    response = await response.text()
                result = re.search(
                    r'<div id="USERID" style="display: none">(.*?)<\/div>',
                    response,
                )

                if result:
                    cls._user_id = result.group(1)
                else:
                    raise RuntimeError("No user id found")
            async with session.post(f"{cls.url}/chat/new_chat", json={"user_id": cls._user_id}, proxy=proxy) as response:
                response.raise_for_status()
                chat_id = (await response.json())["id_"]
            if not chat_id:
                raise RuntimeError("Could not create new chat")
            prompt = format_prompt(messages)
            data = {
                "question": prompt,
                "chat_id": chat_id,
                "timestamp": int(time.time() * 1e3),
            }
            async with session.post(f"{cls.url}/chat/chat_api_stream", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line.startswith(b"data: "):
                        
                        content = json.loads(line[6:])["choices"][0]["delta"].get("content")
                        if content:
                            yield content
                        
            async with session.post(f"{cls.url}/chat/delete_chat", json={"chat_id": chat_id}, proxy=proxy) as response:
                response.raise_for_status()