from __future__ import annotations

import json
import asyncio
from aiohttp import ClientSession, TCPConnector
from urllib.parse import urlencode

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class Feedough(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.feedough.com"
    api_endpoint = "/wp-admin/admin-ajax.php"
    working = True
    default_model = ''

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
            "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
            "dnt": "1",
            "origin": cls.url,
            "referer": f"{cls.url}/ai-prompt-generator/",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        }

        connector = TCPConnector(ssl=False)

        async with ClientSession(headers=headers, connector=connector) as session:
            data = {
                "action": "aixg_generate",
                "prompt": format_prompt(messages),
                "aixg_generate_nonce": "110c021031"
            }

            try:
                async with session.post(
                    f"{cls.url}{cls.api_endpoint}",
                    data=urlencode(data),
                    proxy=proxy
                ) as response:
                    response.raise_for_status()
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                        if response_json.get("success") and "data" in response_json:
                            message = response_json["data"].get("message", "")
                            yield message
                    except json.JSONDecodeError:
                        yield response_text
            except Exception as e:
                print(f"An error occurred: {e}")

    @classmethod
    async def run(cls, *args, **kwargs):
        async for item in cls.create_async_generator(*args, **kwargs):
            yield item

        tasks = asyncio.all_tasks()
        for task in tasks:
            if not task.done():
                await task
