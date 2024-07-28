from __future__ import annotations

import json
from aiohttp import ClientSession, ClientResponseError

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class Marsyoo(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://aiagent.marsyoo.com"
    api_endpoint = "/api/chat-messages"
    working = True
    supports_gpt_4 = True
    default_model = 'gpt-4o'

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "DNT": "1",
            "Origin": cls.url,
            "Referer": f"{cls.url}/chat",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI0MWNkOTE3MS1mNTg1LTRjMTktOTY0Ni01NzgxMTBjYWViNTciLCJzdWIiOiJXZWIgQVBJIFBhc3Nwb3J0IiwiYXBwX2lkIjoiNDFjZDkxNzEtZjU4NS00YzE5LTk2NDYtNTc4MTEwY2FlYjU3IiwiYXBwX2NvZGUiOiJMakhzdWJqNjhMTXZCT0JyIiwiZW5kX3VzZXJfaWQiOiI4YjE5YjY2Mi05M2E1LTRhYTktOGNjNS03MDhmNWE0YmQxNjEifQ.pOzdQ4wTrQjjRlEv1XY9TZitkW5KW1K-wbcUJAoBJ5I",
            "content-type": "application/json",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "Linux",
        }
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "response_mode": "streaming",
                "query": prompt,
                "inputs": {},
            }
            try:
                async with session.post(f"{cls.url}{cls.api_endpoint}", json=data, proxy=proxy) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        if line:
                            try:
                                json_data = json.loads(line.decode('utf-8').strip().lstrip('data: '))
                                if json_data['event'] == 'message':
                                    yield json_data['answer']
                                elif json_data['event'] == 'message_end':
                                    return
                            except json.JSONDecodeError:
                                continue
            except ClientResponseError as e:
                yield f"Error: HTTP {e.status}: {e.message}"
