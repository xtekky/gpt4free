from __future__ import annotations

import time
import hashlib
import uuid

from ..typing import AsyncResult, Messages
from ..requests import StreamSession, raise_for_status
from ..errors import RateLimitError
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

class ChatForAi(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://chatforai.store"
    working = True
    default_model = "gpt-3.5-turbo"
    supports_message_history = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        temperature: float = 0.7,
        top_p: float = 1,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        headers = {
            "Content-Type": "text/plain;charset=UTF-8",
            "Origin": cls.url,
            "Referer": f"{cls.url}/?r=b",
        }
        async with StreamSession(impersonate="chrome", headers=headers, proxies={"https": proxy}, timeout=timeout) as session:
            timestamp = int(time.time() * 1e3)
            conversation_id = str(uuid.uuid4())
            data = {
                "conversationId": conversation_id,
                "conversationType": "chat_continuous",
                "botId": "chat_continuous",
                "globalSettings":{
                    "baseUrl": "https://api.openai.com",
                    "model": model,
                    "messageHistorySize": 5,
                    "temperature": temperature,
                    "top_p": top_p,
                    **kwargs
                },
                "prompt": "",
                "messages": messages,
                "timestamp": timestamp,
                "sign": generate_signature(timestamp, "", conversation_id)
            }
            async with session.post(f"{cls.url}/api/handle/provider-openai", json=data) as response:
                await raise_for_status(response)
                async for chunk in response.iter_content():
                    if b"https://chatforai.store" in chunk:
                        raise RuntimeError(f"Response: {chunk.decode(errors='ignore')}")
                    yield chunk.decode(errors="ignore")

    
def generate_signature(timestamp: int, message: str, id: str):
    buffer = f"{id}:{timestamp}:{message}:h496Jd6b"
    return hashlib.sha256(buffer.encode()).hexdigest()
