from __future__ import annotations

import json

from curl_cffi.requests import AsyncSession

from ..typing import AsyncGenerator
from .base_provider import AsyncGeneratorProvider, format_prompt


class You(AsyncGeneratorProvider):
    url = "https://you.com"
    working = True
    supports_gpt_35_turbo = True
    supports_stream = False


    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        **kwargs,
    ) -> AsyncGenerator:
        async with AsyncSession(proxies={"https": proxy}, impersonate="chrome107") as session:
            headers = {
                "Accept": "text/event-stream",
                "Referer": "https://you.com/search?fromSearchBar=true&tbm=youchat",
            }
            response = await session.get(
                "https://you.com/api/streamingSearch",
                params={"q": format_prompt(messages), "domain": "youchat", "chat": ""},
                headers=headers
            )
            response.raise_for_status()
            start = 'data: {"youChatToken": '
            for line in response.text.splitlines():
                if line.startswith(start):
                    yield json.loads(line[len(start): -1])