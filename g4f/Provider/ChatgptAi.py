from __future__ import annotations

import re
from aiohttp import ClientSession

from ..typing import Messages
from .base_provider import AsyncProvider, format_prompt


class ChatgptAi(AsyncProvider):
    url: str = "https://chatgpt.ai/"
    working = True
    supports_gpt_35_turbo  = True
    _nonce = None
    _post_id = None
    _bot_id = None

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> str:
        headers = {
            "authority"          : "chatgpt.ai",
            "accept"             : "*/*",
            "accept-language"    : "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "cache-control"      : "no-cache",
            "origin"             : "https://chatgpt.ai",
            "pragma"             : "no-cache",
            "referer"            : cls.url,
            "sec-ch-ua"          : '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
            "sec-ch-ua-mobile"   : "?0",
            "sec-ch-ua-platform" : '"Windows"',
            "sec-fetch-dest"     : "empty",
            "sec-fetch-mode"     : "cors",
            "sec-fetch-site"     : "same-origin",
            "user-agent"         : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        }
        async with ClientSession(
            headers=headers
        ) as session:
            if not cls._nonce:
                async with session.get(cls.url, proxy=proxy) as response:
                    response.raise_for_status()
                    text = await response.text()
                result = re.search(r'data-nonce="(.*?)"', text)
                if result:
                    cls._nonce = result.group(1)
                result = re.search(r'data-post-id="(.*?)"', text)
                if result:
                    cls._post_id = result.group(1)
                result = re.search(r'data-bot-id="(.*?)"', text)
                if result:
                    cls._bot_id = result.group(1)
                if not cls._nonce or not cls._post_id or not cls._bot_id:
                    raise RuntimeError("Nonce, post-id or bot-id not found")

            data = {
                "_wpnonce": cls._nonce,
                "post_id": cls._post_id,
                "url": "https://chatgpt.ai",
                "action": "wpaicg_chat_shortcode_message",
                "message": format_prompt(messages),
                "bot_id": cls._bot_id
            }
            async with session.post(
                "https://chatgpt.ai/wp-admin/admin-ajax.php",
                proxy=proxy,
                data=data
            ) as response:
                response.raise_for_status()
                return (await response.json())["data"]