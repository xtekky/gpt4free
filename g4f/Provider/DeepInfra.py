from __future__ import annotations

import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider


class DeepInfra(AsyncGeneratorProvider):
    url = "https://deepinfra.com"
    supports_message_history = True
    working = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = "meta-llama/Llama-2-70b-chat-hf"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/118.0",
            "Accept": "text/event-stream",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{cls.url}/",
            "Content-Type": "application/json",
            "X-Deepinfra-Source": "web-page",
            "Origin": cls.url,
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "model": model,
                "messages": messages,
                "stream": True,
            }
            async with session.post(
                "https://api.deepinfra.com/v1/openai/chat/completions",
                json=data,
                proxy=proxy
            ) as response:
                response.raise_for_status()
                first = True
                async for line in response.content:
                    if line.startswith(b"data: [DONE]"):
                        break
                    elif line.startswith(b"data: "):
                        chunk = json.loads(line[6:])["choices"][0]["delta"].get("content")
                        if chunk:
                            if first:
                                chunk = chunk.lstrip()
                                if chunk:
                                    first = False
                            yield chunk