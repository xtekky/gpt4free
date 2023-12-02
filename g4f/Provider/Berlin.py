from __future__ import annotations

import secrets
import uuid
import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from .helper import format_prompt


class Berlin(AsyncGeneratorProvider):
    url = "https://ai.berlin4h.top"
    working = False
    supports_gpt_35_turbo = True
    _token = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = "gpt-3.5-turbo"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Accept": "*/*",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{cls.url}/",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Alt-Used": "ai.berlin4h.top",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
            "TE": "trailers",
        }
        async with ClientSession(headers=headers) as session:
            if not cls._token:
                data = {
                    "account": '免费使用GPT3.5模型@163.com',
                    "password": '659e945c2d004686bad1a75b708c962f'
                }
                async with session.post(f"{cls.url}/api/login", json=data, proxy=proxy) as response:
                    response.raise_for_status()
                    cls._token = (await response.json())["data"]["token"]
            headers = {
                "token": cls._token
            }
            prompt = format_prompt(messages)
            data = {
                "prompt": prompt,
                "parentMessageId": str(uuid.uuid4()),
                "options": {
                    "model": model,
                    "temperature": 0,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "max_tokens": 1888,
                    **kwargs
                },
            }
            async with session.post(f"{cls.url}/api/chat/completions", json=data, proxy=proxy, headers=headers) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk.strip():
                        try:
                            yield json.loads(chunk)["content"]
                        except:
                            raise RuntimeError(f"Response: {chunk.decode()}")
