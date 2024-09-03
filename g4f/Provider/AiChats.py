from __future__ import annotations

from aiohttp import ClientSession
import json

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from .helper import format_prompt


class AiChats(AsyncGeneratorProvider):
    url = "https://ai-chats.org"
    api_endpoint = "https://ai-chats.org/chat/send2/"
    working = True
    supports_gpt_4 = True
    supports_message_history = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        
        headers = {
            "accept": "application/json, text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": cls.url,
            "pragma": "no-cache",
            "referer": f"{cls.url}/chat/",
            "sec-ch-ua": '"Chromium";v="127", "Not)A;Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        }
        
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "type": "chat",
                "messagesHistory": [
                    {
                        "from": "you",
                        "content": prompt
                    }
                ]
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                full_response = await response.text()
                
                message = ""
                for line in full_response.split('\n'):
                    if line.startswith('data: ') and line != 'data: ':
                        message += line[6:]
                
                message = message.strip()
                
                yield message
