from __future__ import annotations

from ...typing import Messages
from ..base_provider import AsyncProvider, format_prompt
from ..helper import get_cookies
from ...requests import StreamSession

class Aichat(AsyncProvider):
    url = "https://chat-gpt.org/chat"
    working = False
    supports_gpt_35_turbo = True

    @staticmethod
    async def create_async(
        model: str,
        messages: Messages,
        proxy: str = None, **kwargs) -> str:
        
        cookies = get_cookies('chat-gpt.org') if not kwargs.get('cookies') else kwargs.get('cookies')
        if not cookies:
            raise RuntimeError(
                "g4f.provider.Aichat requires cookies, [refresh https://chat-gpt.org on chrome]"
            )

        headers = {
            'authority': 'chat-gpt.org',
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'content-type': 'application/json',
            'origin': 'https://chat-gpt.org',
            'referer': 'https://chat-gpt.org/chat',
            'sec-ch-ua': '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        }

        async with StreamSession(headers=headers,
                                    cookies=cookies,
                                    timeout=6,
                                    proxies={"https": proxy} if proxy else None,
                                    impersonate="chrome110", verify=False) as session:

            json_data = {
                "message": format_prompt(messages),
                "temperature": kwargs.get('temperature', 0.5),
                "presence_penalty": 0,
                "top_p": kwargs.get('top_p', 1),
                "frequency_penalty": 0,
            }

            async with session.post("https://chat-gpt.org/api/text",
                                    json=json_data) as response:

                response.raise_for_status()
                result = await response.json()

                if not result['response']:
                    raise Exception(f"Error Response: {result}")

                return result["message"]
