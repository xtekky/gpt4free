from __future__ import annotations

import time
from hashlib import sha256

from aiohttp import BaseConnector, ClientSession

from ...errors import RateLimitError
from ...requests import raise_for_status
from ...requests.aiohttp import get_connector
from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin


class Free2GPT(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://chat10.free2gpt.xyz"
    
    working = False
    supports_message_history = True
    
    default_model = 'gemini-1.5-pro'
    models = [default_model, 'gemini-1.5-flash']

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        connector: BaseConnector = None,
        **kwargs,
    ) -> AsyncResult:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Content-Type": "text/plain;charset=UTF-8",
            "Referer": f"{cls.url}/",
            "Origin": cls.url,
        }
        async with ClientSession(
            connector=get_connector(connector, proxy), headers=headers
        ) as session:
            timestamp = int(time.time() * 1e3)
            data = {
                "messages": messages,
                "time": timestamp,
                "pass": None,
                "sign": generate_signature(timestamp, messages[-1]["content"]),
            }
            async with session.post(
                f"{cls.url}/api/generate", json=data, proxy=proxy
            ) as response:
                if response.status == 500:
                    if "Quota exceeded" in await response.text():
                        raise RateLimitError(
                            f"Response {response.status}: Rate limit reached"
                        )
                await raise_for_status(response)
                async for chunk in response.content.iter_any():
                    yield chunk.decode(errors="ignore")


def generate_signature(time: int, text: str, secret: str = ""):
    message = f"{time}:{text}:{secret}"
    return sha256(message.encode()).hexdigest()
