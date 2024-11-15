from __future__ import annotations

from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider
from ...requests import get_args_from_browser
from ...webdriver import WebDriver

class Aura(AsyncGeneratorProvider):
    url = "https://openchat.team"
    working = False

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        temperature: float = 0.5,
        max_tokens: int = 8192,
        webdriver: WebDriver = None,
        **kwargs
    ) -> AsyncResult:
        args = get_args_from_browser(cls.url, webdriver, proxy)
        async with ClientSession(**args) as session:
            new_messages = []
            system_message = []
            for message in messages:
                if message["role"] == "system":
                    system_message.append(message["content"])
                else:
                    new_messages.append(message)
            data = {
                "model": {
                    "id": "openchat_3.6",
                    "name": "OpenChat 3.6 (latest)",
                    "maxLength": 24576,
                    "tokenLimit": max_tokens
                },
                "messages": new_messages,
                "key": "",
                "prompt": "\n".join(system_message),
                "temperature": temperature
            }
            async with session.post(f"{cls.url}/api/chat", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_any():
                    yield chunk.decode(error="ignore")
