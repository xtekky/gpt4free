from __future__ import annotations

import json

from ..requests import StreamSession
from ..typing import AsyncGenerator
from .base_provider import AsyncGeneratorProvider, format_prompt


class You(AsyncGeneratorProvider):
    url = "https://you.com"
    working = True
    supports_gpt_35_turbo = True


    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        timeout: int = 30,
        **kwargs,
    ) -> AsyncGenerator:
        async with StreamSession(proxies={"https": proxy}, impersonate="chrome107", timeout=timeout) as session:
            headers = {
                "Accept": "text/event-stream",
                "Referer": "https://you.com/search?fromSearchBar=true&tbm=youchat",
            }
            async with session.get(
                "https://you.com/api/streamingSearch",
                params={"q": format_prompt(messages), "domain": "youchat", "chat": ""},
                headers=headers
            ) as response:
                response.raise_for_status()
                start = b'data: {"youChatToken": '
                async for line in response.iter_lines():
                    if line.startswith(start):
                        yield json.loads(line[len(start):-1])