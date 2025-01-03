from __future__ import annotations

from aiohttp import ClientSession, ClientTimeout
import json

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin


class DDG(AsyncGeneratorProvider, ProviderModelMixin):
    label = "DuckDuckGo AI Chat"
    url = "https://duckduckgo.com/aichat"
    url_status = "https://duckduckgo.com/duckchat/v1/status"
    api_endpoint = "https://duckduckgo.com/duckchat/v1/chat"
    
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-4o-mini'
    models = [default_model, 'claude-3-haiku-20240307', 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', 'mistralai/Mixtral-8x7B-Instruct-v0.1']
    
    model_aliases = {
        "gpt-4": "gpt-4o-mini",
    }
    model_aliases = {
        "gpt-4": default_model,
        "claude-3-haiku": "claude-3-haiku-20240307",
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1"
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        async with ClientSession(timeout=ClientTimeout(total=30)) as session:
            # Fetch VQD token
            async with session.get(cls.url_status, 
                                 headers={"x-vqd-accept": "1"}) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch VQD token: {response.status}")
                vqd = response.headers.get("x-vqd-4", "")
                if not vqd:
                    raise Exception("Failed to fetch VQD token: Empty token.")

            headers = {
                "Content-Type": "application/json",
                "x-vqd-4": vqd,
            }

            payload = {
                "model": model,
                "messages": messages,
            }

            async with session.post(cls.api_endpoint, headers=headers, json=payload, proxy=proxy) as response:
                response.raise_for_status()
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data:"):
                        try:
                            message = json.loads(line[5:].strip())
                            if "message" in message:
                                yield message["message"]
                        except json.JSONDecodeError:
                            continue
