from __future__ import annotations

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class ClaudeSon(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://claudeson.net"
    api_endpoint = "https://claudeson.net/api/coze/chat"
    working = True

    default_model = 'claude-3.5-sonnet'
    models = [default_model]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:      
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://claudeson.net",
            "referer": "https://claudeson.net/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "textStr": format_prompt(messages),
                "type": "company"
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                async for chunk in response.content:
                    if chunk:
                        yield chunk.decode(errors="ignore")