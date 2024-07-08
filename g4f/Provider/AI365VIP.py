from __future__ import annotations

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class AI365VIP(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://chat.ai365vip.com"
    api_endpoint = "/api/chat"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    default_model = 'gpt-3.5-turbo'
    models = [
        'gpt-3.5-turbo',
        'gpt-4o',
        'claude-3-haiku-20240307',
    ]
    model_aliases = {
        "claude-3-haiku": "claude-3-haiku-20240307",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "dnt": "1",
            "origin": "https://chat.ai365vip.com",
            "priority": "u=1, i",
            "referer": "https://chat.ai365vip.com/en",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        }
        async with ClientSession(headers=headers) as session:
            data = {
				"model": {
					"id": model,
					"name": {
						"gpt-3.5-turbo": "GPT-3.5",
						"claude-3-haiku-20240307": "claude-3-haiku",
						"gpt-4o": "GPT-4O"
					}.get(model, model),
				},
				"messages": [{"role": "user", "content": format_prompt(messages)}],
				"prompt": "You are a helpful assistant.",
			}
            async with session.post(f"{cls.url}{cls.api_endpoint}", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk:
                        yield chunk.decode()
