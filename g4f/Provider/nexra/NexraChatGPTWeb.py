from __future__ import annotations
from aiohttp import ClientSession
from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt
import json

class NexraChatGPTWeb(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra ChatGPT Web"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/gptweb"
    models = ['gptweb']

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "Content-Type": "application/json",
        }

        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            if prompt is None:
                raise ValueError("Prompt cannot be None")

            data = {
                "prompt": prompt,
                "markdown": False
            }

            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                
                full_response = ""
                async for chunk in response.content:
                    if chunk:
                        result = chunk.decode("utf-8").strip()
                        
                        try:
                            json_data = json.loads(result)
                            
                            if json_data.get("status"):
                                full_response = json_data.get("gpt", "")
                            else:
                                full_response = f"Error: {json_data.get('message', 'Unknown error')}"
                        except json.JSONDecodeError:
                            full_response = "Error: Invalid JSON response."

                yield full_response.strip()
