from __future__ import annotations

import json
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt


class NexraChatGPT4o(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra GPT-4o"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/complements"
    models = ['gpt-4o']

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "Content-Type": "application/json"
        }
        async with ClientSession(headers=headers) as session:
            data = {
                "messages": [
                    {'role': 'assistant', 'content': ''},
                    {'role': 'user', 'content': format_prompt(messages)}
                ],
                "markdown": False,
                "stream": True,
                "model": model
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                full_response = ''
                async for line in response.content:
                    if line:
                        messages = line.decode('utf-8').split('\x1e')
                        for message_str in messages:
                            try:
                                message = json.loads(message_str)
                                if message.get('message'):
                                    full_response = message['message']
                                if message.get('finish'):
                                    yield full_response.strip()
                                    return
                            except json.JSONDecodeError:
                                pass
