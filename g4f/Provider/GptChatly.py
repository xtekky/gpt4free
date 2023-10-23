# cloudflare block

from __future__ import annotations

from ..requests     import StreamSession
from ..typing       import Messages
from .base_provider import AsyncProvider
from .helper        import get_cookies


class GptChatly(AsyncProvider):
    url                   = "https://gptchatly.com"
    supports_gpt_35_turbo = True
    supports_gpt_4        = True
    working               = True

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None, cookies: dict = None, **kwargs) -> str:

        cookies = get_cookies('gptchatly.com') if not cookies else cookies
        if not cookies:
            raise RuntimeError(
                "g4f.provider.GptChatly requires cookies, [refresh https://gptchatly.com on chrome]"
            )

        if model.startswith("gpt-4"):
            chat_url = f"{cls.url}/fetch-gpt4-response"
        else:
            chat_url = f"{cls.url}/fetch-response"

        headers = {
            'authority': 'gptchatly.com',
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'content-type': 'application/json',
            'origin': 'https://gptchatly.com',
            'referer': 'https://gptchatly.com/',
            'sec-ch-ua': '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        }

        async with StreamSession(headers=headers, 
                                 proxies={"https": proxy}, cookies=cookies, impersonate='chrome110') as session:
            data = {
                "past_conversations": messages
            }
            async with session.post(chat_url, json=data) as response:
                response.raise_for_status()
                return (await response.json())["chatGPTResponse"]