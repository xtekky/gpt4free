from __future__ import annotations

import json

from ...requests     import StreamSession
from ...typing       import AsyncGenerator
from ..base_provider import AsyncGeneratorProvider, format_prompt

class Komo(AsyncGeneratorProvider):
    url = "https://komo.ai/api/ask"
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        **kwargs
    ) -> AsyncGenerator:
        async with StreamSession(impersonate="chrome107") as session:
            prompt = format_prompt(messages)
            data = {
                "query": prompt,
                "FLAG_URLEXTRACT": "false",
                "token": "",
                "FLAG_MODELA": "1",
            }
            headers = {
                'authority': 'komo.ai',
                'accept': 'text/event-stream',
                'cache-control': 'no-cache',
                'referer': 'https://komo.ai/',
            }
            
            async with session.get(cls.url, params=data, headers=headers) as response:
                response.raise_for_status()
                next = False
                async for line in response.iter_lines():
                    if line == b"event: line":
                        next = True
                    elif next and line.startswith(b"data: "):
                        yield json.loads(line[6:])
                        next = False

