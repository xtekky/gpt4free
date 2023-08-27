from aiohttp import ClientSession
import json

from ..typing import AsyncGenerator
from .base_provider import AsyncGeneratorProvider, format_prompt, get_cookies


class You(AsyncGeneratorProvider):
    url = "https://you.com"
    working = True
    supports_gpt_35_turbo = True
    supports_stream = True

    @staticmethod
    async def create_async_generator(
        model: str,
        messages: list[dict[str, str]],
        cookies: dict = None,
        **kwargs,
    ) -> AsyncGenerator:
        if not cookies:
            cookies = get_cookies("you.com")
        headers = {
            "Accept": "text/event-stream",
            "Referer": "https://you.com/search?fromSearchBar=true&tbm=youchat",
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/116.0"
        }
        async with ClientSession(headers=headers, cookies=cookies) as session:
            async with session.get(
                "https://you.com/api/streamingSearch",
                params={"q": format_prompt(messages), "domain": "youchat", "chat": ""},
            ) as response:  
                start = 'data: {"youChatToken": '
                async for line in response.content:
                    line = line.decode('utf-8')
                    if line.startswith(start):
                        yield json.loads(line[len(start): -2])