from __future__ import annotations

import re
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider
from ..helper import format_prompt


class ChatAiGpt(AsyncGeneratorProvider):
    url                   = "https://chataigpt.org"
    supports_gpt_35_turbo = True
    _nonce                = None
    _post_id              = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/118.0",
            "Accept": "*/*",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Origin": cls.url,
            "Alt-Used": cls.url,
            "Connection": "keep-alive",
            "Referer": cls.url,
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
            "TE": "trailers",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }
        async with ClientSession(headers=headers) as session:
            if not cls._nonce:
                async with session.get(f"{cls.url}/", proxy=proxy) as response:
                    response.raise_for_status()
                    response = await response.text()

                    result = re.search(
                        r'data-nonce=(.*?) data-post-id=([0-9]+)', response
                    )

                    if result:
                        cls._nonce, cls._post_id = result.group(1), result.group(2)
                    else:
                        raise RuntimeError("No nonce found")
            prompt = format_prompt(messages)
            data = {
                "_wpnonce": cls._nonce,
                "post_id": cls._post_id,
                "url": cls.url,
                "action": "wpaicg_chat_shortcode_message",
                "message": prompt,
                "bot_id": 0
            }
            async with session.post(f"{cls.url}/wp-admin/admin-ajax.php", data=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk:
                        yield chunk.decode()