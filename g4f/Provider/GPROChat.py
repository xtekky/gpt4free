from __future__ import annotations

import time
import hashlib
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class GPROChat(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://gprochat.com"
    api_endpoint = "https://gprochat.com/api/generate"
    
    working = True
    supports_stream = True
    supports_message_history = True
    default_model = 'gemini-1.5-pro'

    @staticmethod
    def generate_signature(timestamp: int, message: str) -> str:
        secret_key = "2BC120D4-BB36-1B60-26DE-DB630472A3D8"
        hash_input = f"{timestamp}:{message}:{secret_key}"
        signature = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
        return signature

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
        timestamp = int(time.time() * 1000)
        prompt = format_prompt(messages)
        sign = cls.generate_signature(timestamp, prompt)

        headers = {
            "accept": "*/*",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            "content-type": "text/plain;charset=UTF-8"
        }
        
        data = {
            "messages": [{"role": "user", "parts": [{"text": prompt}]}],
            "time": timestamp,
            "pass": None,
            "sign": sign
        }

        async with ClientSession(headers=headers) as session:
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_any():
                    if chunk:
                        yield chunk.decode()
