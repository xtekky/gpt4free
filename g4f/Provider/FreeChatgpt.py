from __future__ import annotations

import json
from aiohttp import ClientSession, ClientTimeout

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

class FreeChatgpt(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://free.chatgpt.org.uk"
    working = True
    supports_message_history = True
    default_model = "google-gemini-pro"

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
            "Accept": "application/json, text/event-stream",
            "Content-Type":"application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.5",
            "Host":"free.chatgpt.org.uk",
            "Referer":f"{cls.url}/",
            "Origin":f"{cls.url}",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", 
        }
        async with ClientSession(headers=headers, timeout=ClientTimeout(timeout)) as session:
            data = {
                "messages": messages,
                "stream": True,
                "model": cls.get_model(""),
                "temperature": kwargs.get("temperature", 0.5),
                "presence_penalty": kwargs.get("presence_penalty", 0),
                "frequency_penalty": kwargs.get("frequency_penalty", 0),
                "top_p": kwargs.get("top_p", 1)
            }
            async with session.post(f'{cls.url}/api/openai/v1/chat/completions', json=data, proxy=proxy) as response:
                response.raise_for_status()
                started = False
                async for line in response.content:
                    if line.startswith(b"data: [DONE]"):
                        break
                    elif line.startswith(b"data: "):
                        line = json.loads(line[6:])
                        if(line["choices"]==[]):
                            continue
                        chunk = line["choices"][0]["delta"].get("content")
                        if chunk:
                            started = True
                            yield chunk
                if not started:
                    raise RuntimeError("Empty response")