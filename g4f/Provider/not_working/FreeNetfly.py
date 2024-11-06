from __future__ import annotations

import json
import asyncio
from aiohttp import ClientSession, ClientTimeout, ClientError
from typing import AsyncGenerator

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin


class FreeNetfly(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://free.netfly.top"
    api_endpoint = "/api/openai/v1/chat/completions"
    working = False
    default_model = 'gpt-3.5-turbo'
    models = [
        'gpt-3.5-turbo',
        'gpt-4',
    ]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "accept": "application/json, text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "dnt": "1",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        }
        data = {
            "messages": messages,
            "stream": True,
            "model": model,
            "temperature": 0.5,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "top_p": 1
        }
        
        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                async with ClientSession(headers=headers) as session:
                    timeout = ClientTimeout(total=60)
                    async with session.post(f"{cls.url}{cls.api_endpoint}", json=data, proxy=proxy, timeout=timeout) as response:
                        response.raise_for_status()
                        async for chunk in cls._process_response(response):
                            yield chunk
                        return  # If successful, exit the function
            except (ClientError, asyncio.TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise  # If all retries failed, raise the last exception
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    @classmethod
    async def _process_response(cls, response) -> AsyncGenerator[str, None]:
        buffer = ""
        async for line in response.content:
            buffer += line.decode('utf-8')
            if buffer.endswith('\n\n'):
                for subline in buffer.strip().split('\n'):
                    if subline.startswith('data: '):
                        if subline == 'data: [DONE]':
                            return
                        try:
                            data = json.loads(subline[6:])
                            content = data['choices'][0]['delta'].get('content')
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON: {subline}")
                        except KeyError:
                            print(f"Unexpected JSON structure: {data}")
                buffer = ""
        
        # Process any remaining data in the buffer
        if buffer:
            for subline in buffer.strip().split('\n'):
                if subline.startswith('data: ') and subline != 'data: [DONE]':
                    try:
                        data = json.loads(subline[6:])
                        content = data['choices'][0]['delta'].get('content')
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError):
                        pass

