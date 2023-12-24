from __future__ import annotations
from aiohttp import ClientSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider

class Aura(AsyncGeneratorProvider):
    url                   = "https://openchat.team"
    working               = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Content-Type": "application/json",
            "Origin": f"{cls.url}",
            "Referer": f"{cls.url}/",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google      Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Linux"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        async with ClientSession(headers=headers) as session:
            new_messages = []
            system_message = []
            for message in messages:
                if message["role"] == "system":
                    system_message.append(message["content"])
                else:
                    new_messages.append(message)
            data = {
                "model": {
                    "id": "openchat_v3.2_mistral",
                    "name": "OpenChat Aura",
                    "maxLength": 24576,
                    "tokenLimit": 8192
                },
                "messages": new_messages,
                "key": "",
                "prompt": "\n".join(system_message),
                "temperature": 0.5
            }
            async with session.post(f"{cls.url}/api/chat", json=data, proxy=proxy) as response:
                async for chunk in response.content.iter_any():
                    yield chunk.decode()