from __future__ import annotations

import os
import re
import json

try:
    from curl_cffi.requests import Session
    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt
from ...errors import MissingRequirementsError

class ChatGptEs(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://chatgpt.es"
    api_endpoint = "https://chatgpt.es/wp-admin/admin-ajax.php"
    
    working = False
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
            "referer": cls.url,
            "origin": cls.url,
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
            "x-requested-with": "XMLHttpRequest",
        })
        
        if proxy:
            session.proxies = {"https": proxy, "http": proxy}
        
        # First request to get nonce and post_id
        initial_response = session.get(cls.url, impersonate="chrome110")
        initial_text = initial_response.text
        
        # More comprehensive nonce extraction
        nonce_patterns = [
            r'<input\s+type=[\'"]hidden[\'"]\s+name=[\'"]_wpnonce[\'"]\s+value=[\'"]([^\'"]+)[\'"]',
            r'"_wpnonce":"([^"]+)"',
            r'var\s+wpaicg_nonce\s*=\s*[\'"]([^\'"]+)[\'"]',
            r'wpaicg_nonce\s*:\s*[\'"]([^\'"]+)[\'"]'
        ]
        
        nonce_ = None
        for pattern in nonce_patterns:
            match = re.search(pattern, initial_text)
            if match:
                nonce_ = match.group(1)
                break
                
        if not nonce_:
            # Try to find any nonce-like pattern as a last resort
            general_nonce = re.search(r'nonce[\'"]?\s*[=:]\s*[\'"]([a-zA-Z0-9]+)[\'"]', initial_text)
            if general_nonce:
                nonce_ = general_nonce.group(1)
            else:
                # Fallback, but this likely won't work
                nonce_ = "8cf9917be2"
        
        # Look for post_id in HTML
        post_id_patterns = [
            r'<input\s+type=[\'"]hidden[\'"]\s+name=[\'"]post_id[\'"]\s+value=[\'"]([^\'"]+)[\'"]',
            r'"post_id":"([^"]+)"',
            r'var\s+post_id\s*=\s*[\'"]?(\d+)[\'"]?'
        ]
        
        post_id = None
        for pattern in post_id_patterns:
            match = re.search(pattern, initial_text)
            if match:
                post_id = match.group(1)
                break
                
        if not post_id:
            post_id = "106"  # Default from curl example
        
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
            if isinstance(result['data'], str) and "Du musst das Kästchen anklicken!" in result['data']:
                raise ValueError(result['data'])
            yield result['data']
        else:
            raise ValueError(f"Unexpected response format: {result}")
