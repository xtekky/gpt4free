from __future__ import annotations

import requests
from ...typing import AsyncResult, Messages
from .OpenaiAPI import OpenaiAPI

class DeepInfra(OpenaiAPI):
    label = "DeepInfra"
    url = "https://deepinfra.com"
    working = True
    needs_auth = True
    supports_stream = True
    supports_message_history = True
    default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    @classmethod
    def get_models(cls):
        if not cls.models:
            url = 'https://api.deepinfra.com/models/featured'
            models = requests.get(url).json()
            cls.models = [model['model_name'] for model in models if model["type"] == "text-generation"]
        return cls.models

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        api_base: str = "https://api.deepinfra.com/v1/openai",
        temperature: float = 0.7,
        max_tokens: int = 1028,
        **kwargs
    ) -> AsyncResult:
        headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US',
            'Connection': 'keep-alive',
            'Origin': 'https://deepinfra.com',
            'Referer': 'https://deepinfra.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'X-Deepinfra-Source': 'web-embed',
            'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
        }
        return super().create_async_generator(
            model, messages,
            stream=stream,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            headers=headers,
            **kwargs
        )
