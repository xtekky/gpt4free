#cloudflare block

from __future__ import annotations

import re
from aiohttp import ClientSession

from ..typing import Messages
from .base_provider import AsyncProvider
from .helper import format_prompt, get_cookies


class ChatgptFree(AsyncProvider):
    url                   = "https://chatgptfree.ai"
    supports_gpt_35_turbo = True
    working               = False
    _post_id              = None
    _nonce                = None

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> str:
        cookies = get_cookies('chatgptfree.ai')
        
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/118.0",
            "Accept": "*/*",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Origin": cls.url,
            "Alt-Used": "chatgptfree.ai",
            "Connection": "keep-alive",
            "Referer": f"{cls.url}/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
            "TE": "trailers"
        }
        async with ClientSession(headers=headers) as session:
            if not cls._nonce:
                async with session.get(f"{cls.url}/", 
                                       proxy=proxy, cookies=cookies) as response:
                    response.raise_for_status()
                    response = await response.text()
                    result = re.search(r'data-post-id="([0-9]+)"', response)
                    if not result:
                        raise RuntimeError("No post id found")
                    cls._post_id = result.group(1)
                    result = re.search(r'data-nonce="(.*?)"', response)
                    if not result:
                        raise RuntimeError("No nonce found")
                    cls._nonce = result.group(1)
            prompt = format_prompt(messages)
            data = {
                "_wpnonce": cls._nonce,
                "post_id": cls._post_id,
                "url": cls.url,
                "action": "wpaicg_chat_shortcode_message",
                "message": prompt,
                "bot_id": "0"
            }
            async with session.post(cls.url + "/wp-admin/admin-ajax.php", data=data, proxy=proxy) as response:
                response.raise_for_status()
                return (await response.json())["data"]