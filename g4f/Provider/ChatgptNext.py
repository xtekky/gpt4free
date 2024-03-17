from __future__ import annotations

import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider

class ChatgptNext(AsyncGeneratorProvider):
    url = "https://www.chatgpt-free.cc"
    working = True
    supports_gpt_35_turbo = True
    supports_message_history = True
    supports_system_message = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        max_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = "gpt-3.5-turbo"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
            "Accept": "text/event-stream",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Content-Type": "application/json",
            "Referer": "https://chat.fstha.com/",
            "x-requested-with": "XMLHttpRequest",
            "Origin": "https://chat.fstha.com",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Authorization": "Bearer ak-chatgpt-nice",
            "Connection": "keep-alive",
            "Alt-Used": "chat.fstha.com",
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "messages": messages,
                "stream": True,
                "model": model,
                "temperature": temperature,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            async with session.post(f"https://chat.fstha.com/api/openai/v1/chat/completions", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk.startswith(b"data: [DONE]"):
                        break
                    if chunk.startswith(b"data: "):
                        content = json.loads(chunk[6:])["choices"][0]["delta"].get("content")
                        if content:
                            yield content