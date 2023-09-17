from __future__ import annotations

import json
import uuid

from aiohttp import ClientSession

from ..typing import AsyncGenerator
from .base_provider import AsyncGeneratorProvider

models = {
    "gpt-4": {
        "id": "gpt-4",
        "name": "GPT-4",
        "maxLength": 24000,
        "tokenLimit": 8000,
    },
    "gpt-3.5-turbo": {
        "id": "gpt-3.5-turbo",
        "name": "GPT-3.5",
        "maxLength": 12000,
        "tokenLimit": 4000,
    },
    "gpt-3.5-turbo-16k": {
        "id": "gpt-3.5-turbo-16k",
        "name": "GPT-3.5-16k",
        "maxLength": 48000,
        "tokenLimit": 16000,
    },
}

class Liaobots(AsyncGeneratorProvider):
    url = "https://liaobots.com"
    working = False
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    _auth_code = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        auth: str = None,
        proxy: str = None,
        **kwargs
    ) -> AsyncGenerator:
        model = model if model in models else "gpt-3.5-turbo"
        if proxy and "://" not in proxy:
            proxy = f"http://{proxy}"
        headers = {
            "authority": "liaobots.com",
            "content-type": "application/json",
            "origin": cls.url,
            "referer": cls.url + "/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        }
        async with ClientSession(
            headers=headers
        ) as session:
            auth_code = auth if isinstance(auth, str) else cls._auth_code
            if not auth_code:
                async with session.post(cls.url + "/api/user", proxy=proxy, json={"authcode": ""}) as response:
                    response.raise_for_status()
                    auth_code = cls._auth_code = json.loads(await response.text())["authCode"]
            data = {
                "conversationId": str(uuid.uuid4()),
                "model": models[model],
                "messages": messages,
                "key": "",
                "prompt": "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully.",
            }
            async with session.post(cls.url + "/api/chat", proxy=proxy, json=data, headers={"x-auth-code": auth_code}) as response:
                response.raise_for_status()
                async for stream in response.content.iter_any():
                    if stream:
                        yield stream.decode()


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("proxy", "str"),
            ("auth", "str"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
