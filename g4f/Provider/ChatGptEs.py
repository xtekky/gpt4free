from __future__ import annotations

from ..typing import Messages, CreateResult
from ..providers.base_provider import AbstractProvider, ProviderModelMixin

import requests, os, re, json

class ChatGptEs(AbstractProvider, ProviderModelMixin):
    label = "ChatGptEs"
    working = True
    supports_message_history = True
    supports_system_message = True
    supports_stream = False

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        **kwargs
    ) -> CreateResult:
        
        if model not in [
            'gpt-4o',
            'gpt-4o-mini',
            'chatgpt-4o-latest'
        ]:
            raise ValueError(f"Unsupported model: {model}")

        headers = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'cache-control': 'no-cache',
            'pragma': 'no-cache',
            'priority': 'u=0, i',
            'referer': 'https://www.google.com/',
            'sec-ch-ua': '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'cross-site',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        }

        response = requests.get('https://chatgpt.es/', headers=headers)
        nonce_   = re.findall(r'data-nonce="(.+?)"', response.text)[0]
        post_id  = re.findall(r'data-post-id="(.+?)"', response.text)[0]

        headers = {
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'cache-control': 'no-cache',
            'origin': 'https://chatgpt.es',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://chatgpt.es/',
            'sec-ch-ua': '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        }
        
        conversation_history = [
            "Human: stricly respond in the same language as my prompt, preferably english"
        ]
        
        for message in messages[:-1]:
            if message['role'] == "user":
                conversation_history.append(f"Human: {message['content']}")
            else:
                conversation_history.append(f"AI: {message['content']}")

        payload = {
            '_wpnonce': nonce_,
            'post_id': post_id,
            'url': 'https://chatgpt.es',
            'action': 'wpaicg_chat_shortcode_message',
            'message': messages[-1]['content'],
            'bot_id': '0',
            'chatbot_identity': 'shortcode',
            'wpaicg_chat_client_id': os.urandom(5).hex(),
            'wpaicg_chat_history': json.dumps(conversation_history)
        }

        response = requests.post('https://chatgpt.es/wp-admin/admin-ajax.php', 
                                headers=headers, data=payload).json()
        
        return (response['data'])