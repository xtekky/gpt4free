from __future__ import annotations
import aiohttp
import asyncio
import json
import uuid
from aiohttp import ClientSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class Allyfy(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://allyfy.chat"
    api_endpoint = "https://chatbot.allyfy.chat/api/v1/message/stream/super/chat"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-3.5-turbo'
    models = [default_model]

    @classmethod
    def get_model(cls, model: str) -> str:
        return cls.default_model

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        client_id = str(uuid.uuid4())

        headers = {
            'accept': 'text/event-stream',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'no-cache',
            'content-type': 'application/json;charset=utf-8',
            'origin': cls.url,
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': f"{cls.url}/",
            'referrer': cls.url,
            'sec-ch-ua': '"Not?A_Brand";v="99", "Chromium";v="130"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'
        }

        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "messages": messages,
                "content": prompt,
                "baseInfo": {
                    "clientId": client_id,
                    "pid": "38281",
                    "channelId": "100000",
                    "locale": "en-US",
                    "localZone": 120,
                    "packageName": "com.cch.allyfy.webh",
                }
            }

            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                response_text = await response.text()

                filtered_response = []
                for line in response_text.splitlines():
                    if line.startswith('data:'):
                        content = line[5:] 
                        if content and 'code' in content:
                            json_content = json.loads(content)
                            if json_content['content']:
                                filtered_response.append(json_content['content'])

                final_response = ''.join(filtered_response)
                yield final_response
