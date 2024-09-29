from __future__ import annotations

import json
import aiohttp
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class DDG(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://duckduckgo.com"
    api_endpoint = "https://duckduckgo.com/duckchat/v1/chat"
    working = True
    supports_gpt_4 = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = "gpt-4o-mini"
    models = [
        "gpt-4o-mini",
        "claude-3-haiku-20240307",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ]
    model_aliases = {
        "claude-3-haiku": "claude-3-haiku-20240307",
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1"
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        return cls.model_aliases.get(model, model) if model in cls.model_aliases else cls.default_model

    @classmethod
    async def get_vqd(cls):
        status_url = "https://duckduckgo.com/duckchat/v1/status"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'Accept': 'text/event-stream',
            'x-vqd-accept': '1'
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(status_url, headers=headers) as response:
                    if response.status == 200:
                        return response.headers.get("x-vqd-4")
                    else:
                        print(f"Error: Status code {response.status}")
                        return None
            except Exception as e:
                print(f"Error getting VQD: {e}")
                return None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        conversation: dict = None,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)

        headers = {
            'accept': 'text/event-stream',
            'content-type': 'application/json',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
        }

        vqd = conversation.get('vqd') if conversation else await cls.get_vqd()
        if not vqd:
            raise Exception("Failed to obtain VQD token")

        headers['x-vqd-4'] = vqd

        if conversation:
            message_history = conversation.get('messages', [])
            message_history.append({"role": "user", "content": format_prompt(messages)})
        else:
            message_history = [{"role": "user", "content": format_prompt(messages)}]

        async with ClientSession(headers=headers) as session:
            data = {
                "model": model,
                "messages": message_history
            }

            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            json_str = decoded_line[6:]
                            if json_str == '[DONE]':
                                break
                            try:
                                json_data = json.loads(json_str)
                                if 'message' in json_data:
                                    yield json_data['message']
                            except json.JSONDecodeError:
                                pass
