from __future__ import annotations

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from ..requests.raise_for_status import raise_for_status

class BoringMarketingChat(AsyncGeneratorProvider):
    url                   = "https://chat.boringmarketing.com"
    working               = True
    supports_gpt_35_turbo = True
    supports_gpt_4        = True
    supports_stream       = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        temperature: float = 0.6,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        payload = {
            "model": model,
            "stream": "true",
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        async with ClientSession() as session:
            async with session.post("https://chat.boringmarketing.com/api/openai/chat", json=payload, proxy=proxy) as response:
                await raise_for_status(response)
                async for chunk in response.content.iter_any():
                    if chunk:
                        yield chunk.decode(errors="ignore")
