from __future__ import annotations

import re
from ..requests import StreamSession, raise_for_status
from ..typing import Messages
from .base_provider import AsyncProvider, ProviderModelMixin
from .helper import format_prompt


class Chatgpt4o(AsyncProvider, ProviderModelMixin):
    url = "https://chatgpt4o.one"
    supports_gpt_4 = True
    working = True
    _post_id = None
    _nonce = None
    default_model = 'gpt-4o-mini-2024-07-18'
    models = [
        'gpt-4o-mini-2024-07-18',
    ]
    model_aliases = {
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    }


    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        cookies: dict = None,
        **kwargs
    ) -> str:
        headers = {
            'authority': 'chatgpt4o.one',
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'origin': 'https://chatgpt4o.one',
            'referer': 'https://chatgpt4o.one',
            'sec-ch-ua': '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        }

        async with StreamSession(
            headers=headers,
            cookies=cookies,
            impersonate="chrome",
            proxies={"all": proxy},
            timeout=timeout
        ) as session:

            if not cls._post_id or not cls._nonce:
                async with session.get(f"{cls.url}/") as response:
                    await raise_for_status(response)
                    response_text = await response.text()

                    post_id_match = re.search(r'data-post-id="([0-9]+)"', response_text)
                    nonce_match = re.search(r'data-nonce="(.*?)"', response_text)

                    if not post_id_match:
                        raise RuntimeError("No post ID found")
                    cls._post_id = post_id_match.group(1)

                    if not nonce_match:
                        raise RuntimeError("No nonce found")
                    cls._nonce = nonce_match.group(1)

            prompt = format_prompt(messages)
            data = {
                "_wpnonce": cls._nonce,
                "post_id": cls._post_id,
                "url": cls.url,
                "action": "wpaicg_chat_shortcode_message",
                "message": prompt,
                "bot_id": "0"
            }

            async with session.post(f"{cls.url}/wp-admin/admin-ajax.php", data=data, cookies=cookies) as response:
                await raise_for_status(response)
                response_json = await response.json()
                if "data" not in response_json:
                    raise RuntimeError("Unexpected response structure: 'data' field missing")
                return response_json["data"]
