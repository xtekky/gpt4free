from __future__ import annotations

from aiohttp import ClientSession
import os
import json
import re

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class ChatGptEs(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://chatgpt.es"
    api_endpoint = "https://chatgpt.es/wp-admin/admin-ajax.php"
    working = True
    supports_gpt_4 = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-4o'
    models = ['gpt-4o', 'gpt-4o-mini', 'chatgpt-4o-latest']
    
    model_aliases = {
        "gpt-4o": "chatgpt-4o-latest",
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            return cls.default_model

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

            conversation_history = [
                "Human: strictly respond in the same language as my prompt, preferably English"
            ]

            for message in messages[:-1]:
                if message['role'] == "user":
                    conversation_history.append(f"Human: {message['content']}")
                else:
                    conversation_history.append(f"AI: {message['content']}")

            payload = {
                '_wpnonce': nonce_,
                'post_id': post_id,
                'url': cls.url,
                'action': 'wpaicg_chat_shortcode_message',
                'message': messages[-1]['content'],
                'bot_id': '0',
                'chatbot_identity': 'shortcode',
                'wpaicg_chat_client_id': os.urandom(5).hex(),
                'wpaicg_chat_history': json.dumps(conversation_history)
            }

            async with session.post(cls.api_endpoint, headers=headers, data=payload) as response:
                response.raise_for_status()
                result = await response.json()
                yield result['data']
