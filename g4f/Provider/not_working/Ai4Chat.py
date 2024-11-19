from __future__ import annotations

import json
import re
import logging
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt

logger = logging.getLogger(__name__)

class Ai4Chat(AsyncGeneratorProvider, ProviderModelMixin):
    label = "AI4Chat"
    url = "https://www.ai4chat.co"
    api_endpoint = "https://www.ai4chat.co/generate-response"
    working = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-4'
    models = [default_model]
    
    model_aliases = {}

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
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": "https://www.ai4chat.co",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://www.ai4chat.co/gpt/talkdirtytome",
            "sec-ch-ua": '"Chromium";v="129", "Not=A?Brand";v="8"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
        }
        
        async with ClientSession(headers=headers) as session:
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": format_prompt(messages)
                    }
                ]
            }
            
            try:
                async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                    response.raise_for_status()
                    result = await response.text()
                    
                    json_result = json.loads(result)
                    
                    message = json_result.get("message", "")
                    
                    clean_message = re.sub(r'<[^>]+>', '', message)
                    
                    yield clean_message
            except Exception as e:
                logger.exception("Error while calling AI 4Chat API: %s", e)
                yield f"Error: {e}"
