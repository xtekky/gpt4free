from __future__ import annotations

from aiohttp import ClientSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class Editee(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Editee"
    url = "https://editee.com"
    api_endpoint = "https://editee.com/submit/chatgptfree"
    working = True
    supports_gpt_4 = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'claude'
    models = ['claude', 'gpt4', 'gemini' 'mistrallarge']
    
    model_aliases = {
        "claude-3.5-sonnet": "claude",
        "gpt-4o": "gpt4",
        "gemini-pro": "gemini",
        "mistral-large": "mistrallarge",
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
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Pragma": "no-cache",
            "Priority": "u=1, i",
            "Referer": f"{cls.url}/chat-gpt",
            "Sec-CH-UA": '"Chromium";v="129", "Not=A?Brand";v="8"',
            "Sec-CH-UA-Mobile": '?0',
            "Sec-CH-UA-Platform": '"Linux"',
            "Sec-Fetch-Dest": 'empty',
            "Sec-Fetch-Mode": 'cors',
            "Sec-Fetch-Site": 'same-origin',
            "User-Agent": 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            "X-Requested-With": 'XMLHttpRequest',
        }

        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "user_input": prompt,
                "context": " ",
                "template_id": "",
                "selected_model": model
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                response_data = await response.json()
                yield response_data['text']
