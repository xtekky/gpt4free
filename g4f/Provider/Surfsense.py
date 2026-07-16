from __future__ import annotations

import json
from typing import Any

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..requests import StreamSession

class Surfsense(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Surfsense"
    url = "https://www.surfsense.com"
    api_endpoint = "https://api.surfsense.com/api/v1/public/anon-chat/stream"
    
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-o4-mini-no-login'
    models = ['gpt-o4-mini-no-login', 'gpt-5.4-mini-no-login']
    
    model_aliases = {
        "o4-mini": "gpt-o4-mini-no-login",
        "gpt-4o-mini": "gpt-o4-mini-no-login",
        "gpt-4.5-mini": "gpt-5.4-mini-no-login"
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str | None = None,
        **kwargs: Any
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        }
        
        data_payload = {
            "model_slug": model,
            "messages": messages
        }

        async with StreamSession(headers=headers, impersonate="safari15_3") as session:
            async with session.post(cls.api_endpoint, json=data_payload, proxy=proxy) as response:
                response.raise_for_status()
                
                async for chunk in response.iter_lines():
                    if not chunk:
                        continue
                    chunk = chunk.decode("utf-8")
                    if chunk.startswith("data: "):
                        chunk = chunk[6:]
                        if chunk == "[DONE]":
                            break
                        try:
                            json_data = json.loads(chunk)
                            if json_data.get("type") == "text-delta":
                                delta = json_data.get("delta")
                                if delta:
                                    yield delta
                        except json.JSONDecodeError:
                            pass
