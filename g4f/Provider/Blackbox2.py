from __future__ import annotations

import random
import asyncio
from aiohttp import ClientSession
from typing import AsyncGenerator

from ..typing import AsyncResult, Messages
from ..image import ImageResponse
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

from .. import debug

class Blackbox2(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.blackbox.ai"
    api_endpoints = {
        "llama-3.1-70b": "https://www.blackbox.ai/api/improve-prompt",
        "flux": "https://www.blackbox.ai/api/image-generator"
    }
    
    working = True
    supports_system_message = True
    supports_message_history = True
    supports_stream = False
       
    default_model = 'llama-3.1-70b'
    chat_models = ['llama-3.1-70b']
    image_models = ['flux']
    models = [*chat_models, *image_models]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        max_retries: int = 3,
        delay: int = 1,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = cls.default_model
        if model in cls.chat_models:
            async for result in cls._generate_text(model, messages, proxy, max_retries, delay):
                yield result
        elif model in cls.image_models:
            prompt = messages[-1]["content"] if prompt is None else prompt
            async for result in cls._generate_image(model, prompt, proxy):
                yield result
        else:
            raise ValueError(f"Unsupported model: {model}")

    @classmethod
    async def _generate_text(
        cls, 
        model: str, 
        messages: Messages, 
        proxy: str = None, 
        max_retries: int = 3, 
        delay: int = 1
    ) -> AsyncGenerator:
        headers = cls._get_headers()
        api_endpoint = cls.api_endpoints[model]

        data = {
            "messages": messages,
            "max_tokens": None
        }

        async with ClientSession(headers=headers) as session:
            for attempt in range(max_retries):
                try:
                    async with session.post(api_endpoint, json=data, proxy=proxy) as response:
                        response.raise_for_status()
                        response_data = await response.json()
                        if 'prompt' in response_data:
                            yield response_data['prompt']
                            return
                        else:
                            raise KeyError("'prompt' key not found in the response")
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Error after {max_retries} attempts: {str(e)}")
                    else:
                        wait_time = delay * (2 ** attempt) + random.uniform(0, 1)
                        debug.log(f"Attempt {attempt + 1} failed. Retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)

    @classmethod
    async def _generate_image(
        cls, 
        model: str, 
        prompt: str, 
        proxy: str = None
    ) -> AsyncGenerator:
        headers = cls._get_headers()
        api_endpoint = cls.api_endpoints[model]

        async with ClientSession(headers=headers) as session:
            data = {
                "query": prompt
            }
            
            async with session.post(api_endpoint, headers=headers, json=data, proxy=proxy) as response:
                response.raise_for_status()
                response_data = await response.json()
                
                if 'markdown' in response_data:
                    image_url = response_data['markdown'].split('(')[1].split(')')[0]
                    yield ImageResponse(images=image_url, alt=prompt)

    @staticmethod
    def _get_headers() -> dict:
        return {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'text/plain;charset=UTF-8',
            'origin': 'https://www.blackbox.ai',
            'priority': 'u=1, i',
            'referer': 'https://www.blackbox.ai',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }
