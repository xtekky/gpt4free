from __future__ import annotations

import json
import logging
from aiohttp import ClientSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

"""
    Mhystical.cc
    ~~~~~~~~~~~~
    Author: NoelP.dev
    Last Updated: 2024-05-11
    
    Author Site: https://noelp.dev
    Provider Site: https://mhystical.cc

"""

logger = logging.getLogger(__name__)

class Mhystical(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://api.mhystical.cc"
    api_endpoint = "https://api.mhystical.cc/v1/completions"
    working = True
    supports_stream = False  # Set to False, as streaming is not specified in ChatifyAI
    supports_system_message = False
    supports_message_history = True
    
    default_model = 'gpt-4'
    models = [default_model]
    model_aliases = {}

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases.get(model, cls.default_model)
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
            "x-api-key": "mhystical",
            "Content-Type": "application/json",
            "accept": "*/*",
            "cache-control": "no-cache",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
        }
        
        async with ClientSession(headers=headers) as session:
            data = {
                "model": model,
                "messages": [{"role": "user", "content": format_prompt(messages)}]
            }
            async with session.post(cls.api_endpoint, json=data, headers=headers, proxy=proxy) as response:
                if response.status == 400:
                    yield "Error: API key is missing"
                elif response.status == 429:
                    yield "Error: Rate limit exceeded"
                elif response.status == 500:
                    yield "Error: Internal server error"
                else:
                    response.raise_for_status()
                    response_text = await response.text()
                    filtered_response = cls.filter_response(response_text)
                    yield filtered_response

    @staticmethod
    def filter_response(response_text: str) -> str:
        try:
            json_response = json.loads(response_text)
            message_content = json_response["choices"][0]["message"]["content"]
            return message_content
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error("Error parsing response: %s", e)
            return "Error: Failed to parse response from API."
