from __future__ import annotations

from ..typing import AsyncResult, Messages
from ..requests import StreamSession
from .base_provider import AsyncGeneratorProvider


class ChatForAi(AsyncGeneratorProvider):
    url                   = "https://chatforai.com"
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
        async with StreamSession(impersonate="chrome107", proxies={"https": proxy}, timeout=timeout) as session:
            prompt = messages[-1]["content"]
            data = {
                "conversationId": "temp",
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
            }
            async with session.post(f"{cls.url}/api/handle/provider-openai", json=data) as response:
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