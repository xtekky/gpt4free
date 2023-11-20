from __future__ import annotations

import re
from aiohttp import ClientSession

from ..typing import Messages
from .base_provider import AsyncProvider
from .helper import format_prompt

class Chatgpt4Online(AsyncProvider):
    url = "https://chatgpt4online.org"
    supports_message_history = True
    supports_gpt_35_turbo = True
    working = True
    _wpnonce = None

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> str:
        async with ClientSession() as session:
            if not cls._wpnonce:
                async with session.get(f"{cls.url}/", proxy=proxy) as response:
                    response.raise_for_status()
                    response = await response.text()
                    result = re.search(r'data-nonce="(.*?)"', response)

                    if result:
                        cls._wpnonce = result.group(1)
                    else:
                        raise RuntimeError("No nonce found")
            data = {
                "_wpnonce": cls._wpnonce,
                "post_id": 58,
                "url": "https://chatgpt4online.org",
                "action": "wpaicg_chat_shortcode_message",
                "message": format_prompt(messages),
                "bot_id": 3405
            }
            async with session.post(f"{cls.url}/rizq", data=data, proxy=proxy) as response:
                response.raise_for_status()
                return (await response.json())["data"]