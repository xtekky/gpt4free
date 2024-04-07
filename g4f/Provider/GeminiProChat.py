from __future__ import annotations

import time
from hashlib import sha256
from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from ..errors import RateLimitError
from ..requests import raise_for_status
from ..requests.aiohttp import get_connector

class GeminiProChat(AsyncGeneratorProvider):
    url = "https://gemini-chatbot-sigma.vercel.app"
    working = True
    supports_message_history = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        connector: BaseConnector = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Content-Type": "text/plain;charset=UTF-8",
            "Referer": "https://gemini-chatbot-sigma.vercel.app/",
            "Origin": "https://gemini-chatbot-sigma.vercel.app",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Connection": "keep-alive",
            "TE": "trailers",
        }
        async with ClientSession(connector=get_connector(connector, proxy), headers=headers) as session:
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
                if response.status == 500:
                    if "Quota exceeded" in await response.text():
                        raise RateLimitError(f"Response {response.status}: Rate limit reached")
                await raise_for_status(response)
                async for chunk in response.content.iter_any():
                    yield chunk.decode(errors="ignore")
                        
def generate_signature(time: int, text: str, secret: str = ""):
    message = f'{time}:{text}:{secret}';
    return sha256(message.encode()).hexdigest()
    
