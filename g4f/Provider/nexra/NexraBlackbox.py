from __future__ import annotations

import json
from aiohttp import ClientSession, ClientTimeout, ClientError

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin

class NexraBlackbox(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra Blackbox"
    url = "https://nexra.aryahcr.cc/documentation/blackbox/en"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/complements"
    working = True
    supports_stream = True
    
    default_model = 'blackbox'
    models = [default_model]
    
    model_aliases = {
        "blackboxai": "blackbox",
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
        stream: bool = False,
        markdown: bool = False,
        websearch: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [{"role": msg["role"], "content": msg["content"]} for msg in messages],
            "websearch": websearch,
            "stream": stream,
            "markdown": markdown,
            "model": model
        }

        timeout = ClientTimeout(total=600)  # 10 minutes timeout
        
        try:
            async with ClientSession(headers=headers, timeout=timeout) as session:
                async with session.post(cls.api_endpoint, json=payload, proxy=proxy) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error: {response.status} - {error_text}")

                    content = await response.text()

                    # Split content by Record Separator character
                    parts = content.split('\x1e')
                    full_message = ""
                    links = []

                    for part in parts:
                        if part:
                            try:
                                json_response = json.loads(part)
                                
                                if json_response.get("message"):
                                    full_message = json_response["message"]  # Overwrite instead of append
                                
                                if isinstance(json_response.get("search"), list):
                                    links = json_response["search"]  # Overwrite instead of extend
                                
                                if json_response.get("finish", False):
                                    break

                            except json.JSONDecodeError:
                                pass

                    if full_message:
                        yield full_message.strip()

                    if payload["websearch"] and links:
                        yield "\n\n**Source:**"
                        for i, link in enumerate(links, start=1):
                            yield f"\n{i}. {link['title']}: {link['link']}"

        except ClientError:
            raise
        except Exception:
            raise
