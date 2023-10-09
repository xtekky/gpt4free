from __future__ import annotations

import time
import hashlib

from ..typing import AsyncResult, Messages
from ..requests import StreamSession
from .base_provider import AsyncGeneratorProvider


class Aibn(AsyncGeneratorProvider):
    url                   = "https://aibn.cc"
    supports_gpt_35_turbo = True
    working               = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        **kwargs
    ) -> AsyncResult:
        async with StreamSession(
            impersonate="chrome107",
            proxies={"https": proxy},
            timeout=timeout
        ) as session:
            timestamp = int(time.time())
            data = {
                "messages": messages,
                "pass": None,
                "sign": generate_signature(timestamp, messages[-1]["content"]),
                "time": timestamp
            }
            async with session.post(f"{cls.url}/api/generate", json=data) as response:
                response.raise_for_status()
                async for chunk in response.iter_content():
                    yield chunk.decode()

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("temperature", "float"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
    

def generate_signature(timestamp: int, message: str, secret: str = "undefined"):
    data = f"{timestamp}:{message}:{secret}"
    return hashlib.sha256(data.encode()).hexdigest()