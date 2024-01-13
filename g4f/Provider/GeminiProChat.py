from __future__ import annotations

import time
from hashlib import sha256
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider


class GeminiProChat(AsyncGeneratorProvider):
    url = "https://geminiprochat.com"
    working = True
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
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Accept": "*/*",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Content-Type": "text/plain;charset=UTF-8",
            "Referer": "https://geminiprochat.com/",
            "Origin": "https://geminiprochat.com",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Connection": "keep-alive",
            "TE": "trailers",
        }
        async with ClientSession(headers=headers) as session:
            timestamp = int(time.time() * 1e3)
            data = {
                "messages":[{
                    "role": "model" if message["role"] == "assistant" else "user",
                    "parts": [{"text": message["content"]}]
                } for message in messages],
                "time": timestamp,
                "pass": None,
                "sign": generate_signature(timestamp, messages[-1]["content"]),
            }
            async with session.post(f"{cls.url}/api/generate", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_any():
                    yield chunk.decode()
                        
def generate_signature(time: int, text: str):
    message = f'{time}:{text}:9C4680FB-A4E1-6BC7-052A-7F68F9F5AD1F';
    return sha256(message.encode()).hexdigest()
