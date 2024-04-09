from __future__ import annotations

from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages, Cookies
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider
from .helper import get_cookies, get_connector, get_random_string

class WhiteRabbitNeo(AsyncGeneratorProvider):
    url = "https://www.whiterabbitneo.com"
    working = True
    supports_message_history = True
    needs_auth = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        cookies: Cookies = None,
        connector: BaseConnector = None,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if cookies is None:
            cookies = get_cookies("www.whiterabbitneo.com")
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
            "Accept": "*/*",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{cls.url}/",
            "Content-Type": "text/plain;charset=UTF-8",
            "Origin": cls.url,
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "TE": "trailers"
        }
        async with ClientSession(
            headers=headers,
            cookies=cookies,
            connector=get_connector(connector, proxy)
        ) as session:
            data = {
                "messages": messages,
                "id": get_random_string(6),
                "enhancePrompt": False,
                "useFunctions": False
            }
            async with session.post(f"{cls.url}/api/chat", json=data, proxy=proxy) as response:
                await raise_for_status(response)
                async for chunk in response.content.iter_any():
                    if chunk:
                        yield chunk.decode(errors="ignore")