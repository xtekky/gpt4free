from __future__ import annotations

from aiohttp import ClientSession
import json
import random
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..typing import AsyncResult, Messages
from .helper import format_prompt

class Bixin123(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://chat.bixin123.com"
    api_endpoint = "https://chat.bixin123.com/api/chatgpt/chat-process"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True

    default_model = 'gpt-3.5-turbo-0125'
    models = ['gpt-3.5-turbo-0125', 'gpt-3.5-turbo-16k-0613', 'gpt-4-turbo', 'qwen-turbo']
    
    model_aliases = {
        "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo": "gpt-3.5-turbo-16k-0613",
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
    def generate_fingerprint(cls) -> str:
        return str(random.randint(100000000, 999999999))

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
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "fingerprint": cls.generate_fingerprint(),
            "origin": cls.url,
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": f"{cls.url}/chat",
            "sec-ch-ua": '"Chromium";v="127", "Not)A;Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "x-website-domain": "chat.bixin123.com",
        }

        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "prompt": prompt,
                "options": {
                    "usingNetwork": False,
                    "file": ""
                }
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                response_text = await response.text()
                
                lines = response_text.strip().split("\n")
                last_json = None
                for line in reversed(lines):
                    try:
                        last_json = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        pass
                
                if last_json:
                    text = last_json.get("text", "")
                    yield text
                else:
                    yield ""
