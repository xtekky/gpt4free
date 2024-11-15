from __future__ import annotations

from aiohttp import ClientSession
import json

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class Upstage(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://console.upstage.ai/playground/chat"
    api_endpoint = "https://ap-northeast-2.apistage.ai/v1/web/demo/chat/completions"
    working = True
    default_model = 'solar-pro'
    models = [
        'upstage/solar-1-mini-chat',
        'upstage/solar-1-mini-chat-ja',
        'solar-pro',
    ]
    model_aliases = {
        "solar-mini": "upstage/solar-1-mini-chat",
        "solar-mini": "upstage/solar-1-mini-chat-ja",
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
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)

        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://console.upstage.ai",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://console.upstage.ai/",
            "sec-ch-ua": '"Not?A_Brand";v="99", "Chromium";v="130"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "cross-site",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        }

        async with ClientSession(headers=headers) as session:
            data = {
                "stream": True,
                "messages": [{"role": "user", "content": format_prompt(messages)}],
                "model": model
            }

            async with session.post(f"{cls.api_endpoint}", json=data, proxy=proxy) as response:
                response.raise_for_status()

                response_text = ""

                async for line in response.content:
                    if line:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                data = json.loads(line[6:])
                                content = data['choices'][0]['delta'].get('content', '')
                                if content:
                                    response_text += content
                                    yield content
                            except json.JSONDecodeError:
                                continue
                        
                        if line == "data: [DONE]":
                            break
