from __future__ import annotations

from aiohttp import ClientSession
import json

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt


class NexraChatGptV2(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Nexra ChatGPT v2"
    url = "https://nexra.aryahcr.cc/documentation/chatgpt/en"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/complements"
    working = True
    supports_gpt_4 = True
    supports_stream = True
    
    default_model = 'chatgpt'
    models = [default_model]

    model_aliases = {
        "gpt-4": "chatgpt",
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
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "Content-Type": "application/json"
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
                "stream": stream,
                "markdown": markdown,
                "model": model
            }

            async with session.post(f"{cls.api_endpoint}", json=data, proxy=proxy) as response:
                response.raise_for_status()

                if stream:
                    # Streamed response handling (stream=True)
                    collected_message = ""
                    async for chunk in response.content.iter_any():
                        if chunk:
                            decoded_chunk = chunk.decode().strip().split("\x1e")
                            for part in decoded_chunk:
                                if part:
                                    message_data = json.loads(part)
                                    
                                    # Collect messages until 'finish': true
                                    if 'message' in message_data and message_data['message']:
                                        collected_message = message_data['message']
                                    
                                    # When finish is true, yield the final collected message
                                    if message_data.get('finish', False):
                                        yield collected_message
                                        return
                else:
                    # Non-streamed response handling (stream=False)
                    response_data = await response.json(content_type=None)
                    
                    # Yield the message directly from the response
                    if 'message' in response_data and response_data['message']:
                        yield response_data['message']
                        return
