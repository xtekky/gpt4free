# cloudflare block

from __future__ import annotations

from aiohttp import ClientSession

from ..typing import Messages
from .base_provider import AsyncProvider
from .helper import get_cookies


class GptChatly(AsyncProvider):
    url                   = "https://gptchatly.com"
    supports_gpt_35_turbo = True
    supports_gpt_4        = True
    working               = False

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None, cookies: dict = None, **kwargs) -> str:

        if not cookies:
            cookies = get_cookies('gptchatly.com')

        
        if model.startswith("gpt-4"):
            chat_url = f"{cls.url}/fetch-gpt4-response"
        else:
            chat_url = f"{cls.url}/fetch-response"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/118.0",
            "Accept": "*/*",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{cls.url}/",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
            "TE": "trailers",
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "past_conversations": messages
            }
            async with session.post(chat_url, json=data, proxy=proxy) as response:
                response.raise_for_status()
                return (await response.json())["chatGPTResponse"]