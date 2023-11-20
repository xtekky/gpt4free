from __future__ import annotations

import time, hashlib, random

from ..typing import AsyncResult, Messages
from ..requests import StreamSession
from .base_provider import AsyncGeneratorProvider

domains = [
    'https://s.aifree.site'
]

class FreeGpt(AsyncGeneratorProvider):
    url = "https://freegpts1.aifree.site/"
    working = False
    supports_message_history = True
    supports_gpt_35_turbo = True

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
                    chunk = chunk.decode()
                    if chunk == "当前地区当日额度已消耗完":
                        raise RuntimeError("Rate limit reached")
                    yield chunk

    
def generate_signature(timestamp: int, message: str, secret: str = ""):
    data = f"{timestamp}:{message}:{secret}"
    return hashlib.sha256(data.encode()).hexdigest()
