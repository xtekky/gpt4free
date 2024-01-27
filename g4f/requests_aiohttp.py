from __future__ import annotations

from aiohttp import ClientSession, ClientResponse, ClientTimeout
from typing import AsyncGenerator, Any

from .Provider.helper import get_connector

class StreamResponse(ClientResponse):
    async def iter_lines(self) -> AsyncGenerator[bytes, None]:
        async for line in self.content:
            yield line.rstrip(b"\r\n")

    async def json(self) -> Any:
        return await super().json(content_type=None)

class StreamSession(ClientSession):
    def __init__(self, headers: dict = {}, timeout: int = None, proxies: dict = {}, impersonate = None, **kwargs):
        if impersonate:
            headers = {
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'en-US',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-site',
                "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
                'Accept': '*/*',
                'sec-ch-ua': '"Google Chrome";v="107", "Chromium";v="107", "Not?A_Brand";v="24"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                **headers
            }
        super().__init__(
            **kwargs,
            timeout=ClientTimeout(timeout) if timeout else None,
            response_class=StreamResponse,
            connector=get_connector(kwargs.get("connector"), proxies.get("https")),
            headers=headers
        )