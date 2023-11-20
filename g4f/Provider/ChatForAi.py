from __future__ import annotations

import time
import hashlib

from ..typing import AsyncResult, Messages
from ..requests import StreamSession
from .base_provider import AsyncGeneratorProvider


class ChatForAi(AsyncGeneratorProvider):
    url = "https://chatforai.store"
    working = True
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
        headers = {
            "Content-Type": "text/plain;charset=UTF-8",
            "Origin": cls.url,
            "Referer": f"{cls.url}/?r=b",
        }
        async with StreamSession(impersonate="chrome107", headers=headers, proxies={"https": proxy}, timeout=timeout) as session:
            prompt = messages[-1]["content"]
            timestamp = int(time.time() * 1e3)
            conversation_id = f"id_{timestamp-123}"
            data = {
                "conversationId": conversation_id,
                "conversationType": "chat_continuous",
                "botId": "chat_continuous",
                "globalSettings":{
                    "baseUrl": "https://api.openai.com",
                    "model": model if model else "gpt-3.5-turbo",
                    "messageHistorySize": 5,
                    "temperature": 0.7,
                    "top_p": 1,
                    **kwargs
                },
                "botSettings": {},
                "prompt": prompt,
                "messages": messages,
                "timestamp": timestamp,
                "sign": generate_signature(timestamp, prompt, conversation_id)
            }
            async with session.post(f"{cls.url}/api/handle/provider-openai", json=data) as response:
                response.raise_for_status()
                async for chunk in response.iter_content():
                    if b"https://chatforai.store" in chunk:
                        raise RuntimeError(f"Response: {chunk.decode()}")
                    yield chunk.decode()

    
def generate_signature(timestamp: int, message: str, id: str):
    buffer = f"{timestamp}:{id}:{message}:7YN8z6d6"
    return hashlib.sha256(buffer.encode()).hexdigest()
