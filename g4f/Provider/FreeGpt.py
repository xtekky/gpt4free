from __future__ import annotations

import time, hashlib, random

from ..typing import AsyncResult, Messages
from ..requests import StreamSession, raise_for_status
from .base_provider import AsyncGeneratorProvider
from ..errors import RateLimitError

domains = [
    "https://s.aifree.site",
    "https://v.aifree.site/"
]

class FreeGpt(AsyncGeneratorProvider):
    url = "https://freegptsnav.aifree.site"
    working = True
    supports_message_history = True
    supports_system_message = True
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
            impersonate="chrome",
            timeout=timeout,
            proxies={"all": proxy}
        ) as session:
            prompt = messages[-1]["content"]
            timestamp = int(time.time())
            data = {
                "messages": messages,
                "time": timestamp,
                "pass": None,
                "sign": generate_signature(timestamp, prompt)
            }
            domain = random.choice(domains)
            async with session.post(f"{domain}/api/generate", json=data) as response:
                await raise_for_status(response)
                async for chunk in response.iter_content():
                    chunk = chunk.decode(errors="ignore")
                    if chunk == "当前地区当日额度已消耗完":
                        raise RateLimitError("Rate limit reached")
                    yield chunk
    
def generate_signature(timestamp: int, message: str, secret: str = ""):
    data = f"{timestamp}:{message}:{secret}"
    return hashlib.sha256(data.encode()).hexdigest()
