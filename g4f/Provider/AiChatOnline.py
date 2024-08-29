from __future__ import annotations

import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_random_string, format_prompt

class AiChatOnline(AsyncGeneratorProvider, ProviderModelMixin):
    site_url = "https://aichatonline.org"
    url = "https://aichatonlineorg.erweima.ai"
    api_endpoint = "/aichatonline/api/chat/gpt"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    default_model = 'gpt-4o-mini'
    supports_message_history = False

    @classmethod
    async def grab_token(
        cls,
        session: ClientSession,
        proxy: str
    ):
        async with session.get(f'https://aichatonlineorg.erweima.ai/api/v1/user/getUniqueId?canvas=-{get_random_string()}', proxy=proxy) as response:
            response.raise_for_status()
            return (await response.json())['data']
        
    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{cls.url}/chatgpt/chat/",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Alt-Used": "aichatonline.org",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "TE": "trailers"
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "conversationId": get_random_string(),
                "prompt": format_prompt(messages),
            }
            headers['UniqueId'] = await cls.grab_token(session, proxy)
            async with session.post(f"{cls.url}{cls.api_endpoint}", headers=headers, json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    try:
                        yield json.loads(chunk)['data']['message']
                    except:
                        continue