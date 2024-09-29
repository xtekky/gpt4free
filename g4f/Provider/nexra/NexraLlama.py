from __future__ import annotations

import json
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt


class NexraLlama(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra LLaMA 3.1"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/complements"
    models = ['llama-3.1']

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
