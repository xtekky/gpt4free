from __future__ import annotations
from aiohttp import ClientSession
from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt
import json

class NexraBing(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra Bing"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/complements"

    bing_models = {
        'Bing (Balanced)': 'Balanced',
        'Bing (Creative)': 'Creative',
        'Bing (Precise)': 'Precise'
    }
    
    models = [*bing_models.keys()]

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
            "Accept": "application/json",
            "Origin": cls.url or "https://default-url.com",
            "Referer": f"{cls.url}/chat" if cls.url else "https://default-url.com/chat",
        }

        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            if prompt is None:
                raise ValueError("Prompt cannot be None")

            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "conversation_style": cls.bing_models.get(model, 'Balanced'),
                "markdown": False,
                "stream": True,
                "model": "Bing"
            }

            full_response = ""
            last_message = ""

            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                
                async for line in response.content:
                    if line:
                        raw_data = line.decode('utf-8').strip()

                        parts = raw_data.split('')
                        for part in parts:
                            if part:
                                try:
                                    json_data = json.loads(part)
                                except json.JSONDecodeError:
                                    continue

                                if json_data.get("error"):
                                    raise Exception("Error in API response")

                                if json_data.get("finish"):
                                    break

                                if message := json_data.get("message"):
                                    if message != last_message:
                                        full_response = message
                                        last_message = message

            yield full_response.strip()
