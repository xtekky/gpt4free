from __future__ import annotations

from aiohttp import ClientSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider

class AiAsk(AsyncGeneratorProvider):
    url = "https://e.aiask.me"
    supports_gpt_35_turbo = True
    working = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "accept": "application/json, text/plain, */*",
            "origin": cls.url,
            "referer": f"{cls.url}/chat",
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "continuous": True,
                "id": "fRMSQtuHl91A4De9cCvKD",
                "list": messages,
                "models": "0",
                "prompt": "",
                "temperature": kwargs.get("temperature", 0.5),
                "title": "",
            }
            buffer = ""
            rate_limit = "您的免费额度不够使用这个模型啦，请点击右上角登录继续使用！"
            async with session.post(f"{cls.url}/v1/chat/gpt/", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_any():
                    buffer += chunk.decode()
                    if not rate_limit.startswith(buffer):
                        yield buffer
                        buffer = ""
                    elif buffer == rate_limit:
                        raise RuntimeError("Rate limit reached")