from __future__ import annotations

import json
import requests

from ...typing import CreateResult, Messages
from ..base_provider import ProviderModelMixin, AbstractProvider
from ..helper import format_prompt

class NexraBing(AbstractProvider, ProviderModelMixin):
    label = "Nexra Bing"
    url = "https://nexra.aryahcr.cc/documentation/bing/en"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/complements"
    working = True
    supports_stream = True
    
    default_model = 'Balanced'
    models = [default_model, 'Creative', 'Precise']
    
    model_aliases = {
        "gpt-4": "Balanced",
        "gpt-4": "Creative",
        "gpt-4": "Precise",
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
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
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
            "conversation_style": model,
            "markdown": markdown,
            "stream": stream,
            "model": "Bing"
        }
        
        response = requests.post(cls.api_endpoint, headers=headers, json=data, stream=True)

        return cls.process_response(response)

    @classmethod
    def process_response(cls, response):
        if response.status_code != 200:
            yield f"Error: {response.status_code}"
            return

        full_message = ""
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                messages = chunk.decode('utf-8').split('\x1e')
                for message in messages:
                    try:
                        json_data = json.loads(message)
                        if json_data.get('finish', False):
                            return
                        current_message = json_data.get('message', '')
                        if current_message:
                            new_content = current_message[len(full_message):]
                            if new_content:
                                yield new_content
                                full_message = current_message
                    except json.JSONDecodeError:
                        continue

        if not full_message:
            yield "No message received"
