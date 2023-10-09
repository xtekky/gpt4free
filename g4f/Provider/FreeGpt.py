from __future__ import annotations

import time, hashlib, random

from ..typing import AsyncResult, Messages
from ..requests import StreamSession
from .base_provider import AsyncGeneratorProvider

domains = [
    'https://k.aifree.site',
    'https://p.aifree.site'
]

class FreeGpt(AsyncGeneratorProvider):
    url                   = "https://freegpts1.aifree.site/"
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
            timeout=timeout,
            proxies={"https": proxy}
        ) as session:
            prompt = messages[-1]["content"]
            timestamp = int(time.time())
            data = {
                "messages": messages,
                "time": timestamp,
                "pass": None,
                "sign": generate_signature(timestamp, prompt)
            }
            url = random.choice(domains)
            async with session.post(f"{url}/api/generate", json=data) as response:
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
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
    
def generate_signature(timestamp: int, message: str, secret: str = ""):
    data = f"{timestamp}:{message}:{secret}"
    return hashlib.sha256(data.encode()).hexdigest()