from __future__ import annotations

import json
import requests

from ...typing import CreateResult, Messages
from ..base_provider import ProviderModelMixin, AbstractProvider
from ..helper import format_prompt

class NexraBlackbox(AbstractProvider, ProviderModelMixin):
    label = "Nexra Blackbox"
    url = "https://nexra.aryahcr.cc/documentation/blackbox/en"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/complements"
    working = True
    supports_stream = True
    
    default_model = "blackbox"
    models = [default_model]
    model_aliases = {"blackboxai": "blackbox",}

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
        websearch: bool = False,
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
            "websearch": websearch,
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
                full_response = ""
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        data = json.loads(line)
                        if data.get('finish'):
                            break
                        message = data.get('message', '')
                        if message:
                            full_response = message
                return full_response
            except json.JSONDecodeError:
                return "Error: Unable to decode JSON response"
        else:
            return f"Error: {response.status_code}"

    @classmethod
    def process_streaming_response(cls, response):
        previous_message = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    data = json.loads(line)
                    if data.get('finish'):
                        break
                    message = data.get('message', '')
                    if message and message != previous_message:
                        yield message[len(previous_message):]
                        previous_message = message
                except json.JSONDecodeError:
                    pass
