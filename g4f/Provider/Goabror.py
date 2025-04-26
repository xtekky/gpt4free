from __future__ import annotations

import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..requests.raise_for_status import raise_for_status
from .helper import format_prompt, get_system_prompt

class Goabror(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://goabror.uz"
    api_endpoint = "https://goabror.uz/api/gpt.php"
    working = True

    default_model = 'gpt-4'
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
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-language': 'en-US,en;q=0.9',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
        }
        async with ClientSession(headers=headers) as session:
            params = {
                "user": format_prompt(messages, include_system=False),
                "system": get_system_prompt(messages),
            }
            async with session.get(f"{cls.api_endpoint}", params=params, proxy=proxy) as response:
                await raise_for_status(response)
                text_response = await response.text()
                try:
                    json_response = json.loads(text_response)
                    if "data" in json_response:
                        yield json_response["data"]
                    else:
                        yield text_response
                except json.JSONDecodeError:
                    yield text_response
