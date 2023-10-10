from __future__ import annotations

import re
import json

from aiohttp import ClientSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from .helper import format_prompt


class ChatgptX(AsyncGeneratorProvider):
    url = "https://chatgptx.de"
    supports_gpt_35_turbo = True
    working               = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            'accept-language': 'de-DE,de;q=0.9,en-DE;q=0.8,en;q=0.7,en-US',
            'sec-ch-ua': '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': 'Linux',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
        }
        async with ClientSession(headers=headers) as session:
            async with session.get(f"{cls.url}/", proxy=proxy) as response:
                response = await response.text()
                result = re.search(r'<meta name="csrf-token" content="(.*?)"', response)
                if result:
                    csrf_token = result.group(1)
                result = re.search(r"openconversions\('(.*?)'\)", response)
                if result:
                    chat_id = result.group(1)
                result = re.search(r'<input type="hidden" id="user_id" value="(.*?)"', response)
                if result:
                    user_id = result.group(1)

            if not csrf_token or not chat_id or not user_id:
                raise RuntimeError("Missing csrf_token, chat_id or user_id")

            data = {
                '_token': csrf_token,
                'user_id': user_id,
                'chats_id': chat_id,
                'prompt': format_prompt(messages),
                'current_model': "gpt3"
            }
            headers = {
                'authority': 'chatgptx.de',
                'accept': 'application/json, text/javascript, */*; q=0.01',
                'origin': cls.url,
                'referer': f'{cls.url}/',
                'x-csrf-token': csrf_token,
                'x-requested-with': 'XMLHttpRequest'
            }
            async with session.post(cls.url + '/sendchat', data=data, headers=headers, proxy=proxy) as response:
                response.raise_for_status()
                chat = await response.json()
                if "response" not in chat or not chat["response"]:
                    raise RuntimeError(f'Response: {chat}')
            headers = {
                'authority': 'chatgptx.de',
                'accept': 'text/event-stream',
                'referer': f'{cls.url}/',
                'x-csrf-token': csrf_token,
                'x-requested-with': 'XMLHttpRequest'
            }
            data = {
                "user_id": user_id,
                "chats_id": chat_id,
                "prompt": format_prompt(messages),
                "current_model": "gpt3",
                "conversions_id": chat["conversions_id"],
                "ass_conversions_id": chat["ass_conversions_id"],
            }
            async with session.get(f'{cls.url}/chats_stream', params=data, headers=headers, proxy=proxy) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line.startswith(b"data: "):
                        row = line[6:-1]
                        if row == b"[DONE]":
                            break
                        try:
                            content = json.loads(row)["choices"][0]["delta"].get("content")
                        except:
                            raise RuntimeError(f"Broken line: {line.decode()}")
                        if content:
                            yield content