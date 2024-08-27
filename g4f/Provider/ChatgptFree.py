from __future__ import annotations

import re
import json
import asyncio
from ..requests import StreamSession, raise_for_status
from ..typing import Messages, AsyncGenerator
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class ChatgptFree(AsyncGeneratorProvider, ProviderModelMixin):
    url                   = "https://chatgptfree.ai"
    supports_gpt_4 = True
    working               = True
    _post_id              = None
    _nonce                = None
    default_model = 'gpt-4o-mini-2024-07-18'
    model_aliases = {
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        cookies: dict = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        headers = {
            'authority': 'chatgptfree.ai',
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'origin': 'https://chatgptfree.ai',
            'referer': 'https://chatgptfree.ai/chat/',
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

            if not cls._nonce:
                async with session.get(f"{cls.url}/") as response:
                    await raise_for_status(response)
                    response = await response.text()

                    result = re.search(r'data-post-id="([0-9]+)"', response)
                    if not result:
                        raise RuntimeError("No post id found")
                    cls._post_id = result.group(1)

                    result = re.search(r'data-nonce="(.*?)"', response)
                    if result:
                        cls._nonce = result.group(1)
                    else:
                        raise RuntimeError("No nonce found")

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
                buffer = ""
                async for line in response.iter_lines():
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            json_data = json.loads(data)
                            content = json_data['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
                    elif line:
                        buffer += line

                if buffer:
                    try:
                        json_response = json.loads(buffer)
                        if 'data' in json_response:
                            yield json_response['data']
                    except json.JSONDecodeError:
                        print(f"Failed to decode final JSON. Buffer content: {buffer}")
