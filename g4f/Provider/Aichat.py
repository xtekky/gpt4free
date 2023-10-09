from __future__ import annotations

from aiohttp import ClientSession

from ..typing import Messages
from .base_provider import AsyncProvider, format_prompt


class Aichat(AsyncProvider):
    url                   = "https://chat-gpt.org/chat"
    working               = True
    supports_gpt_35_turbo = True

    @staticmethod
    async def create_async(
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> str:
        headers = {
            "authority": "chat-gpt.org",
            "accept": "*/*",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": "https://chat-gpt.org",
            "pragma": "no-cache",
            "referer": "https://chat-gpt.org/chat",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
        }
        async with ClientSession(
            headers=headers
        ) as session:
            json_data = {
                "message": format_prompt(messages),
                "temperature": kwargs.get('temperature', 0.5),
                "presence_penalty": 0,
                "top_p": kwargs.get('top_p', 1),
                "frequency_penalty": 0,
            }
            async with session.post(
                "https://chat-gpt.org/api/text",
                proxy=proxy,
                json=json_data
            ) as response:
                response.raise_for_status()
                result = await response.json()
                if not result['response']:
                    raise Exception(f"Error Response: {result}")
                return result["message"]
