from __future__ import annotations

from ..requests import StreamSession
from .base_provider import AsyncGeneratorProvider
from ..typing import AsyncResult, Messages

# to recreate this easily, send a post request to https://chat.aivvm.com/api/models
models = {
    'gpt-3.5-turbo': {'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5'},
    'gpt-3.5-turbo-0613': {'id': 'gpt-3.5-turbo-0613', 'name': 'GPT-3.5-0613'},
    'gpt-3.5-turbo-16k': {'id': 'gpt-3.5-turbo-16k', 'name': 'GPT-3.5-16K'},
    'gpt-3.5-turbo-16k-0613': {'id': 'gpt-3.5-turbo-16k-0613', 'name': 'GPT-3.5-16K-0613'},
    'gpt-4': {'id': 'gpt-4', 'name': 'GPT-4'},
    'gpt-4-0613': {'id': 'gpt-4-0613', 'name': 'GPT-4-0613'},
    'gpt-4-32k': {'id': 'gpt-4-32k', 'name': 'GPT-4-32K'},
    'gpt-4-32k-0613': {'id': 'gpt-4-32k-0613', 'name': 'GPT-4-32K-0613'},
}

class Aivvm(AsyncGeneratorProvider):
    url                   = 'https://chat.aivvm.com'
    supports_gpt_35_turbo = True
    supports_gpt_4        = True
    working = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        timeout: int = 120,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = "gpt-3.5-turbo"
        elif model not in models:
            raise ValueError(f"Model is not supported: {model}")

        json_data = {
            "model"       : models[model],
            "messages"    : messages,
            "key"         : "",
            "prompt"      : kwargs.get("system_message", "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown."),
            "temperature" : kwargs.get("temperature", 0.7)
        }
        headers = {
            "Accept": "*/*",
            "Origin": cls.url,
            "Referer": f"{cls.url}/",
        }
        async with StreamSession(
            impersonate="chrome107",
            headers=headers,
            proxies={"https": proxy},
            timeout=timeout
        ) as session:
            async with session.post(f"{cls.url}/api/chat", json=json_data) as response:
                response.raise_for_status()
                async for chunk in response.iter_content():
                    if b'Access denied | chat.aivvm.com used Cloudflare' in chunk:
                        raise ValueError("Rate Limit | use another provider")
                    
                    yield chunk.decode()

    @classmethod
    @property
    def params(cls):
        params = [
            ('model', 'str'),
            ('messages', 'list[dict[str, str]]'),
            ('stream', 'bool'),
            ('temperature', 'float'),
        ]
        param = ', '.join([': '.join(p) for p in params])
        return f'g4f.provider.{cls.__name__} supports: ({param})'