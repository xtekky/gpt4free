from __future__ import annotations

import json

from ...requests import StreamSession
from ..base_provider import AsyncGeneratorProvider
from ...typing import AsyncResult, Messages

class Ylokh(AsyncGeneratorProvider):
    url = "https://chat.ylokh.xyz"
    working = False
    supports_message_history = True 
    supports_gpt_35_turbo = True


    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        proxy: str = None,
        timeout: int = 120,
        **kwargs
    ) -> AsyncResult:
        model = model if model else "gpt-3.5-turbo"
        headers = {"Origin": cls.url, "Referer": f"{cls.url}/"}
        data = {
            "messages": messages,
            "model": model,
            "temperature": 1,
            "presence_penalty": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "allow_fallback": True,
            "stream": stream,
            **kwargs
        }
        async with StreamSession(
                headers=headers,
                proxies={"https": proxy},
                timeout=timeout
            ) as session:
            async with session.post("https://chatapi.ylokh.xyz/v1/chat/completions", json=data) as response:
                response.raise_for_status()
                if stream:
                    async for line in response.iter_lines():
                        line = line.decode()
                        if line.startswith("data: "):
                            if line.startswith("data: [DONE]"):
                                break
                            line = json.loads(line[6:])
                            content = line["choices"][0]["delta"].get("content")
                            if content:
                                yield content
                else:
                    chat = await response.json()
                    yield chat["choices"][0]["message"].get("content")