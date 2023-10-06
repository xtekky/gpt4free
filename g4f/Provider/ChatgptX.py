from __future__ import annotations

import re

from aiohttp import ClientSession

from .base_provider import AsyncProvider
from .helper import format_prompt


class ChatgptX(AsyncProvider):
    url = "https://chatgptx.de"
    supports_gpt_35_turbo = True
    working               = True

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: list[dict[str, str]],
        **kwargs
    ) -> str:
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
            async with session.get(f"{cls.url}/") as response:
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
            async with session.post(cls.url + '/sendchat', data=data, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                if "message" in data:
                    return data["message"]
                elif "messages" in data:
                    raise RuntimeError(f'Response: {data["messages"]}')