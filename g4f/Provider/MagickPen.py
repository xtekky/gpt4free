from __future__ import annotations

import time
import random
import hashlib
import re
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class MagickPen(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://magickpen.com"
    api_endpoint_free = "https://api.magickpen.com/chat/free"
    api_endpoint_ask = "https://api.magickpen.com/ask"
    working = True
    supports_gpt_4 = True
    supports_stream = False
    
    default_model = 'free'
    models = ['free', 'ask']
    
    model_aliases = {
        "gpt-4o-mini": "free",
        "gpt-4o-mini": "ask",
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
    async def get_secrets(cls):
        url = 'https://magickpen.com/_nuxt/02c76dc.js'
        async with ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    x_api_secret_match = re.search(r'"X-API-Secret":"([^"]+)"', text)
                    secret_match = re.search(r'secret:\s*"([^"]+)"', text)
                    
                    x_api_secret = x_api_secret_match.group(1) if x_api_secret_match else None
                    secret = secret_match.group(1) if secret_match else None
                    
                    # Generate timestamp and nonce dynamically
                    timestamp = str(int(time.time() * 1000))
                    nonce = str(random.random())
                    
                    # Generate signature
                    signature_parts = ["TGDBU9zCgM", timestamp, nonce]
                    signature_string = "".join(sorted(signature_parts))
                    signature = hashlib.md5(signature_string.encode()).hexdigest()
                    
                    return {
                        'X-API-Secret': x_api_secret,
                        'signature': signature,
                        'timestamp': timestamp,
                        'nonce': nonce,
                        'secret': secret
                    }
                else:
                    print(f"Error while fetching the file: {response.status}")
                    return None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        secrets = await cls.get_secrets()
        if not secrets:
            raise Exception("Failed to obtain necessary secrets")

        headers = {
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "nonce": secrets['nonce'],
            "origin": "https://magickpen.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://magickpen.com/",
            "sec-ch-ua": '"Chromium";v="127", "Not)A;Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "secret": secrets['secret'],
            "signature": secrets['signature'],
            "timestamp": secrets['timestamp'],
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "x-api-secret": secrets['X-API-Secret']
        }
        
        async with ClientSession(headers=headers) as session:
            if model == 'free':
                data = {
                    "history": [{"role": "user", "content": format_prompt(messages)}]
                }
                async with session.post(cls.api_endpoint_free, json=data, proxy=proxy) as response:
                    response.raise_for_status()
                    result = await response.text()
                    yield result
            
            elif model == 'ask':
                data = {
                    "query": format_prompt(messages),
                    "plan": "Pay as you go"
                }
                async with session.post(cls.api_endpoint_ask, json=data, proxy=proxy) as response:
                    response.raise_for_status()
                    async for chunk in response.content:
                        if chunk:
                            yield chunk.decode()
            
            else:
                raise ValueError(f"Unknown model: {model}")
