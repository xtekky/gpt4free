from __future__ import annotations

import os
import re

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class ChatGptt(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://chatgptt.me"
    api_endpoint = "https://chatgptt.me/wp-admin/admin-ajax.php"
    
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-4o'
    models = ['gpt-4', default_model, 'gpt-4o-mini']

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "authority": "chatgptt.me",
            "accept": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/chat",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        }

        async with ClientSession(headers=headers) as session:
            initial_response = await session.get(cls.url)
            nonce_ = re.findall(r'data-nonce="(.+?)"', await initial_response.text())[0]
            post_id = re.findall(r'data-post-id="(.+?)"', await initial_response.text())[0]

            payload = {
                '_wpnonce': nonce_,
                'post_id': post_id,
                'url': cls.url,
                'action': 'wpaicg_chat_shortcode_message',
                'message': format_prompt(messages),
                'bot_id': '0',
                'chatbot_identity': 'shortcode',
                'wpaicg_chat_client_id': os.urandom(5).hex(),
                'wpaicg_chat_history': None
            }

            async with session.post(cls.api_endpoint, headers=headers, data=payload, proxy=proxy) as response:
                await raise_for_status(response)
                result = await response.json()
                yield result['data']