from __future__ import annotations

from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider


class TalkAi(AsyncGeneratorProvider):
    url                   = "https://talkai.info"
    supports_gpt_35_turbo = True

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
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/118.0",
            "Accept": "application/json",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{cls.url}/de/chat/",
            "content-type": "application/json",
            "Origin": cls.url,
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache"
        }
        async with ClientSession(headers=headers) as session:
            history = [{
                "content": message["content"],
                "from": "you" if message["role"] == "user" else "chatGPT"
            } for message in messages]
            data = {
                "type": "chat",
                "message": messages[-1]["content"],
                "messagesHistory": history,
                "model": model,
                "max_tokens": 256,
                "temperature": 1,
                "top_p": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                **kwargs
            }
            async with session.post(f"{cls.url}/de/chat/send2/", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk:
                        yield chunk.decode()