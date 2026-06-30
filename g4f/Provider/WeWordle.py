from __future__ import annotations

import json
from typing import Any

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..requests import StreamSession
from ..providers.response import Reasoning

class WeWordle(AsyncGeneratorProvider, ProviderModelMixin):
    label = "WeWordle"
    url = "https://chat-gpt.com"
    api_endpoint = "https://llmproxy.org/api/chat.php" 
    
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'v3'
    models = [default_model, 'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'deepseek', 'deepseek-reasoner', 'deepseek-r1']
    
    model_aliases = {
        "deepseek": "v3",
        "deepseek-reasoner": "v3",
        "deepseek-r1": "v3"
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str | None = None,
        **kwargs: Any
    ) -> AsyncResult:
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": "https://chat-gpt.com",
            "referer": "https://chat-gpt.com/"
        }
        
        web_search = kwargs.get("web_search", False)
        
        data_payload = {
            "messages": messages,
            "model": cls.get_model(model),
            "cost": 1,
            "stream": True,
            "web_search": web_search
        }

        async with StreamSession(headers=headers, impersonate="chrome") as session:
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
                            if "choices" in json_data and isinstance(json_data["choices"], list) and json_data["choices"]:
                                delta = json_data["choices"][0].get("delta", {})
                                
                                if "reasoning" in delta and delta["reasoning"]:
                                    yield Reasoning(status=delta["reasoning"])
                                    
                                if "content" in delta and delta["content"]:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            pass
