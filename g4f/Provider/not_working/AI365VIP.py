from __future__ import annotations

from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt


class AI365VIP(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://chat.ai365vip.com"
    api_endpoint = "/api/chat"
    working = False
    default_model = 'gpt-3.5-turbo'
    models = [
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-16k',
        'gpt-4o',
    ]
    model_aliases = {
        "gpt-3.5-turbo": "gpt-3.5-turbo-16k",
    }

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
            "origin": cls.url,
            "referer": f"{cls.url}/en",
            "sec-ch-ua": '"Chromium";v="127", "Not)A;Brand";v="99"',
            "sec-ch-ua-arch": '"x86"',
            "sec-ch-ua-bitness": '"64"',
            "sec-ch-ua-full-version": '"127.0.6533.119"',
            "sec-ch-ua-full-version-list": '"Chromium";v="127.0.6533.119", "Not)A;Brand";v="99.0.0.0"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-model": '""',
            "sec-ch-ua-platform": '"Linux"',
            "sec-ch-ua-platform-version": '"4.19.276"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "model": {
                    "id": model,
                    "name": "GPT-3.5",
                    "maxLength": 3000,
                    "tokenLimit": 2048
                },
                "messages": [{"role": "user", "content": format_prompt(messages)}],
                "key": "",
                "prompt": "You are a helpful assistant.",
                "temperature": 1
            }
            async with session.post(f"{cls.url}{cls.api_endpoint}", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk:
                        yield chunk.decode()
