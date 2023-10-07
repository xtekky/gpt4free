from __future__ import annotations

import os, re
from aiohttp import ClientSession

from ..base_provider import AsyncProvider, format_prompt


class ChatgptLogin(AsyncProvider):
    url                   = "https://opchatgpts.net"
    supports_gpt_35_turbo = True
    working               = True
    _nonce                = None

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: list[dict[str, str]],
        **kwargs
    ) -> str:
        headers = {
            "User-Agent"         : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            "Accept"             : "*/*",
            "Accept-language"    : "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "Origin"             : "https://opchatgpts.net",
            "Alt-Used"           : "opchatgpts.net",
            "Referer"            : "https://opchatgpts.net/chatgpt-free-use/",
            "Sec-Fetch-Dest"     : "empty",
            "Sec-Fetch-Mode"     : "cors",
            "Sec-Fetch-Site"     : "same-origin",
        }
        async with ClientSession(
            headers=headers
        ) as session:
            if not cls._nonce:
                async with session.get(
                    "https://opchatgpts.net/chatgpt-free-use/",
                    params={"id": os.urandom(6).hex()},
                ) as response:
                    result = re.search(r'data-nonce="(.*?)"', await response.text())
                if not result:
                    raise RuntimeError("No nonce value")
                cls._nonce = result.group(1)
            data = {
                "_wpnonce": cls._nonce,
                "post_id": 28,
                "url": "https://opchatgpts.net/chatgpt-free-use",
                "action": "wpaicg_chat_shortcode_message",
                "message": format_prompt(messages),
                "bot_id": 0
            }
            async with session.post("https://opchatgpts.net/wp-admin/admin-ajax.php", data=data) as response:
                response.raise_for_status()
                data = await response.json()
                if "data" in data:
                    return data["data"]
                elif "msg" in data:
                    raise RuntimeError(data["msg"])
                else:
                    raise RuntimeError(f"Response: {data}")


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("temperature", "float"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"