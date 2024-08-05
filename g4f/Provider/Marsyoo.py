from __future__ import annotations

import json
import aiohttp
from aiohttp import ClientSession, ClientResponseError

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class Marsyoo(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://aiagent.marsyoo.com"
    api_endpoint = "/api/chat-messages"
    passport_endpoint = "/api/passport"
    working = True
    supports_gpt_4 = True
    default_model = 'gpt-4o'

    @classmethod
    async def get_access_token(cls, proxy: str = None) -> str:
        headers = {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "DNT": "1",
            "Referer": f"{cls.url}/chat/LjHsubj68LMvBOBr",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "authorization": "Bearer",
            "content-type": "application/json",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "x-app-code": "LjHsubj68LMvBOBr"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{cls.url}{cls.passport_endpoint}", headers=headers, proxy=proxy) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('access_token', '')
                else:
                    raise Exception(f"Error: {response.status}")

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        access_token = await cls.get_access_token(proxy)
        
        headers = {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "DNT": "1",
            "Origin": cls.url,
            "Referer": f"{cls.url}/chat",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "authorization": f"Bearer {access_token}",
            "content-type": "application/json",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "Linux",
        }
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "response_mode": "streaming",
                "query": prompt,
                "inputs": {},
            }
            try:
                async with session.post(f"{cls.url}{cls.api_endpoint}", json=data, proxy=proxy) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        if line:
                            try:
                                json_data = json.loads(line.decode('utf-8').strip().lstrip('data: '))
                                if json_data['event'] == 'message':
                                    yield json_data['answer']
                                elif json_data['event'] == 'message_end':
                                    return
                            except json.JSONDecodeError:
                                continue
            except ClientResponseError as e:
                yield f"Error: HTTP {e.status}: {e.message}"
