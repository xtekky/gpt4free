from __future__ import annotations

import json

import requests

from ...typing import CreateResult, Messages
from ..base_provider import AbstractProvider


class Raycast(AbstractProvider):
    url                     = "https://raycast.com"
    supports_gpt_35_turbo   = True
    supports_gpt_4          = True
    supports_stream         = True
    needs_auth              = True
    working                 = True

    @staticmethod
    def create_completion(
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        **kwargs,
    ) -> CreateResult:
        auth = kwargs.get('auth')
        headers = {
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Authorization': f'Bearer {auth}',
            'Content-Type': 'application/json',
            'User-Agent': 'Raycast/0 CFNetwork/1410.0.3 Darwin/22.6.0',
        }
        parsed_messages = [
            {'author': message['role'], 'content': {'text': message['content']}}
            for message in messages
        ]
        data = {
            "debug": False,
            "locale": "en-CN",
            "messages": parsed_messages,
            "model": model,
            "provider": "openai",
            "source": "ai_chat",
            "system_instruction": "markdown",
            "temperature": 0.5
        }
        response = requests.post(
            "https://backend.raycast.com/api/v1/ai/chat_completions",
            headers=headers,
            json=data,
            stream=True,
            proxies={"https": proxy}
        )
        for token in response.iter_lines():
            if b'data: ' not in token:
                continue
            completion_chunk = json.loads(token.decode().replace('data: ', ''))
            token = completion_chunk['text']
            if token != None:
                yield token
