from __future__ import annotations

import os
import re
import json

try:
    from curl_cffi.requests import Session
    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..errors import MissingRequirementsError

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
        if not has_curl_cffi:
            raise MissingRequirementsError('Install or update "curl_cffi" package | pip install -U curl_cffi')
            
        model = cls.get_model(model)
        prompt = f"{cls.SYSTEM_PROMPT} {format_prompt(messages)}"
        
        # Use curl_cffi with automatic Cloudflare bypass
        session = Session()
        session.headers.update({
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        })
        
        if proxy:
            session.proxies = {"https": proxy, "http": proxy}
        
        # First request to get nonce and post_id
        initial_response = session.get(cls.url, impersonate="chrome110")
        initial_text = initial_response.text
        
        # Look for nonce in HTML
        nonce_match = re.search(r'<input\s+type=[\'"]hidden[\'"]\s+name=[\'"]_wpnonce[\'"]\s+value=[\'"]([^\'"]+)[\'"]', initial_text)
        if not nonce_match:
            json_match = re.search(r'"_wpnonce":"([^"]+)"', initial_text)
            if json_match:
                nonce_ = json_match.group(1)
            else:
                # If not found, use default value
                nonce_ = "8cf9917be2"
        else:
            nonce_ = nonce_match.group(1)
        
        # Look for post_id in HTML
        post_id_match = re.search(r'<input\s+type=[\'"]hidden[\'"]\s+name=[\'"]post_id[\'"]\s+value=[\'"]([^\'"]+)[\'"]', initial_text)
        if not post_id_match:
            post_id = "106"  # Default from curl example
        else:
            post_id = post_id_match.group(1)
        
        client_id = os.urandom(5).hex()
        
        # Prepare data
        data = {
            '_wpnonce': nonce_,
            'post_id': post_id,
            'url': cls.url,
            'action': 'wpaicg_chat_shortcode_message',
            'message': prompt,
            'bot_id': '0',
            'chatbot_identity': 'shortcode',
            'wpaicg_chat_client_id': client_id,
            'wpaicg_chat_history': json.dumps([f"Human: {prompt}"])
        }

        # Execute POST request
        response = session.post(
            cls.api_endpoint,
            data=data,
            impersonate="chrome110"
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error: {response.status_code} - {response.text}")
        
        result = response.json()
        if "data" in result:
            if "Du musst das Kästchen anklicken!" in result['data']:
                raise ValueError(result['data'])
            yield result['data']
        else:
            raise ValueError(f"Unexpected response format: {result}")
