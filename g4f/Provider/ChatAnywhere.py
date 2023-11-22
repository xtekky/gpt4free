from __future__ import annotations

from aiohttp import ClientSession, ClientTimeout

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider


class ChatAnywhere(AsyncGeneratorProvider):
    url = "https://chatanywhere.cn"
    supports_gpt_35_turbo = True
    supports_message_history = True
    working = False

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        temperature: float = 0.5,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Content-Type": "application/json",
            "Referer": f"{cls.url}/",
            "Origin": cls.url,
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Authorization": "",
            "Connection": "keep-alive",
            "TE": "trailers"
        }
        async with ClientSession(headers=headers, timeout=ClientTimeout(timeout)) as session:
            data = {
                "list": messages,
                "id": "s1_qYuOLXjI3rEpc7WHfQ",
                "title": messages[-1]["content"],
                "prompt": "",
                "temperature": temperature,
                "models": "61490748",
                "continuous": True
            }
            async with session.post(f"{cls.url}/v1/chat/gpt/", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_any():
                    if chunk:
                        yield chunk.decode()