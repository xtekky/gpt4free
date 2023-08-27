import uuid
import json
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
    supports_stream = True
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
        if proxy and "://" not in proxy:
            proxy = f"http://{proxy}"
        headers = {
            "authority": "liaobots.com",
            "content-type": "application/json",
            "origin": "https://liaobots.com",
            "referer": "https://liaobots.com/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        }
        async with ClientSession(
            headers=headers
        ) as session:
            model = model if model in models else "gpt-3.5-turbo"
            auth_code = auth if isinstance(auth, str) else cls._auth_code
            if not auth_code:
                async with session.post("https://liaobots.com/api/user", proxy=proxy, json={"authcode": ""}) as response:
                    response.raise_for_status()
                    auth_code = cls._auth_code = json.loads((await response.text()))["authCode"]
            data = {
                "conversationId": str(uuid.uuid4()),
                "model": models[model],
                "messages": messages,
                "key": "",
                "prompt": "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully.",
            }
            async with session.post("https://liaobots.com/api/chat", proxy=proxy, json=data, headers={"x-auth-code": auth_code}) as response:
                response.raise_for_status()
                async for line in response.content:
                    yield line.decode("utf-8")


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
