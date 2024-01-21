from __future__ import annotations

import json, random
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider

models = {
    "claude-v2": "claude-2.0",
    "claude-v2.1":"claude-2.1",
    "gemini-pro": "google-gemini-pro"
}
urls = [
    "https://free.chatgpt.org.uk",
    "https://ai.chatgpt.org.uk"
]

class FreeChatgpt(AsyncGeneratorProvider):
    url = "https://free.chatgpt.org.uk"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    supports_message_history = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if model in models:
            model = models[model]
        elif not model:
            model = "gpt-3.5-turbo"
        url = random.choice(urls)
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
        async with ClientSession(headers=headers) as session:
            data = {
                "messages":messages,
                "stream":True,
                "model":model,
                "temperature":0.5,
                "presence_penalty":0,
                "frequency_penalty":0,
                "top_p":1,
                **kwargs
            }
            async with session.post(f'{url}/api/openai/v1/chat/completions', json=data, proxy=proxy) as response:
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