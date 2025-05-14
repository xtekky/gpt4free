from __future__ import annotations

from ...typing import Messages
from ...requests import StreamSession
from ..base_provider import AsyncProvider, format_prompt


class ChatgptDuo(AsyncProvider):
    url                   = "https://chatgptduo.com"
    supports_gpt_35_turbo = True
    working               = False

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        **kwargs
    ) -> str:
        async with StreamSession(
            impersonate="chrome107",
            proxies={"https": proxy},
            timeout=timeout
        ) as session:
            prompt = format_prompt(messages),
            data = {
                "prompt": prompt,
                "search": prompt,
                "purpose": "ask",
            }
            response = await session.post(f"{cls.url}/", data=data)
            response.raise_for_status()
            data = response.json()

            cls._sources = [{
                "title": source["title"],
                "url": source["link"],
                "snippet": source["snippet"]
            } for source in data["results"]]

            return data["answer"]

    @classmethod
    def get_sources(cls):
        return cls._sources