from __future__ import annotations

import json
from aiohttp import ClientSession, ClientTimeout, StreamReader

from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class DarkAI(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://darkai.foundation/chat"
    api_endpoint = "https://darkai.foundation/chat"
    working = True
    supports_stream = True

    default_model = 'llama-3-70b'
    models = [
         'gpt-4o',
         'gpt-3.5-turbo',
         default_model,
    ]
    model_aliases = {
        "llama-3.1-70b": "llama-3-70b",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)

        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
        }
        
        timeout = ClientTimeout(total=600)  # Increase timeout to 10 minutes
        
        async with ClientSession(headers=headers, timeout=timeout) as session:
            prompt = format_prompt(messages)
            data = {
                "query": prompt,
                "model": model,
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                reader: StreamReader = response.content
                buffer = b""
                while True:
                    chunk = await reader.read(1024)  # Read in smaller chunks
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if line:
                            try:
                                line_str = line.decode()
                                if line_str.startswith('data: '):
                                    chunk_data = json.loads(line_str[6:])
                                    if chunk_data['event'] == 'text-chunk':
                                        chunk = chunk_data['data']['text']
                                        yield chunk
                                    elif chunk_data['event'] == 'stream-end':
                                        return
                            except json.JSONDecodeError:
                                pass
                            except Exception:
                                pass
