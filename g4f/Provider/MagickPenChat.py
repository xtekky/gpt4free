from __future__ import annotations

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class MagickPenChat(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://magickpen.com/chat"
    api_endpoint = "https://api.magickpen.com/chat/free"
    working = True
    supports_gpt_4 = True
    default_model = "gpt-4o-mini"
    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9",
            "access-control-allow-origin": "*",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://magickpen.com",
            "priority": "u=1, i",
            "referer": "https://magickpen.com/",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            'X-Api-Secret': 'W252GY255JVYBS9NAM'
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "history": [{"role": "user", "content": format_prompt(messages)}]
            }
            async with session.post(f"{cls.api_endpoint}", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk:
                        yield chunk.decode()
