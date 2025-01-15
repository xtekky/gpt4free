from __future__ import annotations

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..providers.response import FinishReason

class Pizzagpt(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.pizzagpt.it"
    api_endpoint = "/api/chatx-completion"
    
    working = False
    
    default_model = 'gpt-4o-mini'
    models = [default_model]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "accept": "application/json",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/en",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "x-secret": "Marinara"
        }
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "question": prompt
            }
            async with session.post(f"{cls.url}{cls.api_endpoint}", json=data, proxy=proxy) as response:
                response.raise_for_status()
                response_json = await response.json()
                content = response_json.get("answer", response_json).get("content")
                if content:
                    if "Misuse detected. please get in touch" in content:
                        raise ValueError(content)
                    yield content
                    yield FinishReason("stop")
