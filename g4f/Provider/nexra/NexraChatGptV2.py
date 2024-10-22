from __future__ import annotations

import json
import requests

from ...typing import CreateResult, Messages
from ..base_provider import ProviderModelMixin, AbstractProvider
from ..helper import format_prompt

class NexraChatGptV2(AbstractProvider, ProviderModelMixin):
    label = "Nexra ChatGPT v2"
    url = "https://nexra.aryahcr.cc/documentation/chatgpt/en"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/complements"
    working = True
    supports_stream = True
    
    default_model = 'chatgpt'
    models = [default_model]
    model_aliases = {"gpt-4": "chatgpt"}

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            return cls.default_model
            
    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        markdown: bool = False,
        **kwargs
    ) -> CreateResult:
        model = cls.get_model(model)

        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": format_prompt(messages)
                }
            ],
            "stream": stream,
            "markdown": markdown,
            "model": model
        }
        
        response = requests.post(cls.api_endpoint, headers=headers, json=data, stream=stream)

        if stream:
            return cls.process_streaming_response(response)
        else:
            return cls.process_non_streaming_response(response)

    @classmethod
    def process_non_streaming_response(cls, response):
        if response.status_code == 200:
            try:
                content = response.text.lstrip('')
                data = json.loads(content)
                return data.get('message', '')
            except json.JSONDecodeError:
                return "Error: Unable to decode JSON response"
        else:
            return f"Error: {response.status_code}"

    @classmethod
    def process_streaming_response(cls, response):
        full_message = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    line = line.lstrip('')
                    data = json.loads(line)
                    if data.get('finish'):
                        break
                    message = data.get('message', '')
                    if message:
                        yield message[len(full_message):]
                        full_message = message
                except json.JSONDecodeError:
                    pass
