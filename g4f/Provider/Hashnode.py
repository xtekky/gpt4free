from __future__ import annotations

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from .helper import get_random_hex

class SearchTypes():
    quick = "quick"
    code = "code"
    websearch = "websearch"

class Hashnode(AsyncGeneratorProvider):
    url = "https://hashnode.com"
    working = True
    supports_message_history = True
    supports_gpt_35_turbo = True
    _sources = []

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        search_type: str = SearchTypes.websearch,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/118.0",
            "Accept": "*/*",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{cls.url}/rix",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
            "TE": "trailers",
        }
        async with ClientSession(headers=headers) as session:
            prompt = messages[-1]["content"]
            cls._sources = []
            if search_type == "websearch":
                async with session.post(
                    f"{cls.url}/api/ai/rix/search",
                    json={"prompt": prompt},
                    proxy=proxy,
                ) as response:
                    response.raise_for_status()
                    cls._sources = (await response.json())["result"]
            data = {
                "chatId": get_random_hex(),
                "history": messages,
                "prompt": prompt,
                "searchType": search_type,
                "urlToScan": None,
                "searchResults": cls._sources,
            }
            async with session.post(
                f"{cls.url}/api/ai/rix/completion",
                json=data,
                proxy=proxy,
            ) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_any():
                    if chunk:
                        yield chunk.decode()

    @classmethod
    def get_sources(cls) -> list:
        return [{
            "title": source["name"],
            "url": source["url"]
        } for source in cls._sources]