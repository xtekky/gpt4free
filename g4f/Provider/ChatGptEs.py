from __future__ import annotations

import os
import re

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class ChatGptEs(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://chatgpt.es"
    api_endpoint = "https://chatgpt.es/wp-admin/admin-ajax.php"
    
    working = True
    supports_stream = True
    supports_system_message = False
    supports_message_history = False
    
    default_model = 'gpt-4o'
    models = ['gpt-4', default_model, 'gpt-4o-mini']
    
    SYSTEM_PROMPT = "Your default language is English. Always respond in English unless the user's message is in a different language. If the user's message is not in English, respond in the language of the user's message. Maintain this language behavior throughout the conversation unless explicitly instructed otherwise. User input:"

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
            "authority": "chatgpt.es",
            "accept": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/chat",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        }

        async with ClientSession(headers=headers) as session:
            initial_response = await session.get(cls.url)
            nonce_ = re.findall(r'data-nonce="(.+?)"', await initial_response.text())[0]
            post_id = re.findall(r'data-post-id="(.+?)"', await initial_response.text())[0]
            
            prompt = f"{cls.SYSTEM_PROMPT} {format_prompt(messages)}"

            payload = {
                'check_51710191': '1',
                '_wpnonce': nonce_,
                'post_id': post_id,
                'url': cls.url,
                'action': 'wpaicg_chat_shortcode_message',
                'message': prompt,
                'bot_id': '0',
                'chatbot_identity': 'shortcode',
                'wpaicg_chat_client_id': os.urandom(5).hex(),
                'wpaicg_chat_history': None
            }

            async with session.post(cls.api_endpoint, headers=headers, data=payload) as response:
                await raise_for_status(response)
                result = await response.json()
                if "Du musst das Kästchen anklicken!" in result['data']:
                    raise ValueError(result['data'])
                yield result['data']
