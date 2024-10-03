from __future__ import annotations

from aiohttp import ClientSession
import hashlib
import time
import random
import re
import json
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class MagickPen(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://magickpen.com"
    api_endpoint = "https://api.magickpen.com/ask"
    working = True
    supports_gpt_4 = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-4o-mini'
    models = ['gpt-4o-mini']

    @classmethod
    async def fetch_api_credentials(cls) -> tuple:
        url = "https://magickpen.com/_nuxt/bf709a9ce19f14e18116.js"
        async with ClientSession() as session:
            async with session.get(url) as response:
                text = await response.text()

        pattern = r'"X-API-Secret":"(\w+)"'
        match = re.search(pattern, text)
        X_API_SECRET = match.group(1) if match else None

        timestamp = str(int(time.time() * 1000))
        nonce = str(random.random())

        s = ["TGDBU9zCgM", timestamp, nonce]
        s.sort()
        signature_string = ''.join(s)
        signature = hashlib.md5(signature_string.encode()).hexdigest()

        pattern = r'secret:"(\w+)"'
        match = re.search(pattern, text)
        secret = match.group(1) if match else None

        if X_API_SECRET and timestamp and nonce and secret:
            return X_API_SECRET, signature, timestamp, nonce, secret
        else:
            raise Exception("Unable to extract all the necessary data from the JavaScript file.")

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        X_API_SECRET, signature, timestamp, nonce, secret = await cls.fetch_api_credentials()
        
        headers = {
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'nonce': nonce,
            'origin': cls.url,
            'referer': f"{cls.url}/",
            'secret': secret,
            'signature': signature,
            'timestamp': timestamp,
            'x-api-secret': X_API_SECRET,
        }
        
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            payload = {
                'query': prompt,
                'turnstileResponse': '',
                'action': 'verify'
            }
            async with session.post(cls.api_endpoint, json=payload, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk:
                        yield chunk.decode()
