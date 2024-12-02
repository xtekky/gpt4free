from __future__ import annotations

import random
import asyncio
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

class Blackbox2(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.blackbox.ai"
    api_endpoint = "https://www.blackbox.ai/api/improve-prompt"
    working = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'llama-3.1-70b'
    models = [default_model]

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
        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'text/plain;charset=UTF-8',
            'dnt': '1',
            'origin': 'https://www.blackbox.ai',
            'priority': 'u=1, i',
            'referer': 'https://www.blackbox.ai',
            'sec-ch-ua': '"Chromium";v="131", "Not_A Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }

        data = {
            "messages": messages,
            "max_tokens": None
        }

        async with ClientSession(headers=headers) as session:
            for attempt in range(max_retries):
                try:
                    async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                        response.raise_for_status()
                        response_data = await response.json()
                        if 'prompt' in response_data:
                            yield response_data['prompt']
                            return
                        else:
                            raise KeyError("'prompt' key not found in the response")
                except Exception as e:
                    if attempt == max_retries - 1:
                        yield f"Error after {max_retries} attempts: {str(e)}"
                    else:
                        wait_time = delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Attempt {attempt + 1} failed. Retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
