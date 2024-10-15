from __future__ import annotations

from aiohttp import ClientSession
from aiohttp.client_exceptions import ContentTypeError

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt
import json


class NexraBing(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra Bing"
    url = "https://nexra.aryahcr.cc/documentation/bing/en"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/complements"
    working = False
    supports_gpt_4 = False
    supports_stream = False
    
    default_model = 'Bing (Balanced)'
    models = ['Bing (Balanced)', 'Bing (Creative)', 'Bing (Precise)']
    
    model_aliases = {
        "gpt-4": "Bing (Balanced)",
        "gpt-4": "Bing (Creative)",
        "gpt-4": "Bing (Precise)",
    }

    @classmethod
    def get_model_and_style(cls, model: str) -> tuple[str, str]:
        # Default to the default model if not found
        model = cls.model_aliases.get(model, model)
        if model not in cls.models:
            model = cls.default_model

        # Extract the base model and conversation style
        base_model, conversation_style = model.split(' (')
        conversation_style = conversation_style.rstrip(')')
        return base_model, conversation_style

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        stream: bool = False,
        markdown: bool = False,
        **kwargs
    ) -> AsyncResult:
        base_model, conversation_style = cls.get_model_and_style(model)
        
        headers = {
            "Content-Type": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/chat",
        }
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "conversation_style": conversation_style,
                "markdown": markdown,
                "stream": stream,
                "model": base_model
            }
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                try:
                    # Read the entire response text
                    text_response = await response.text()
                    # Split the response on the separator character
                    segments = text_response.split('\x1e')
                    
                    complete_message = ""
                    for segment in segments:
                        if not segment.strip():
                            continue
                        try:
                            response_data = json.loads(segment)
                            if response_data.get('message'):
                                complete_message = response_data['message']
                            if response_data.get('finish'):
                                break
                        except json.JSONDecodeError:
                            raise Exception(f"Failed to parse segment: {segment}")

                    # Yield the complete message
                    yield complete_message
                except ContentTypeError:
                    raise Exception("Failed to parse response content type.")
