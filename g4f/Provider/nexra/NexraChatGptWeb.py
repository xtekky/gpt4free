from __future__ import annotations

import json
import requests

from ...typing import CreateResult, Messages
from ..base_provider import ProviderModelMixin, AbstractProvider
from ..helper import format_prompt

class NexraChatGptWeb(AbstractProvider, ProviderModelMixin):
    label = "Nexra ChatGPT Web"
    url = "https://nexra.aryahcr.cc/documentation/chatgpt/en"
    working = True
    
    default_model = "gptweb"
    models = [default_model]
    model_aliases = {"gpt-4": "gptweb"}
    api_endpoints = {"gptweb": "https://nexra.aryahcr.cc/api/chat/gptweb"}

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
        proxy: str = None,
        markdown: bool = False,
        **kwargs
    ) -> CreateResult:
        model = cls.get_model(model)
        api_endpoint = cls.api_endpoints.get(model, cls.api_endpoints[cls.default_model])

        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            "prompt": format_prompt(messages),
            "markdown": markdown
        }
        
        response = requests.post(api_endpoint, headers=headers, json=data)

        return cls.process_response(response)

    @classmethod
    def process_response(cls, response):
        if response.status_code == 200:
            try:
                content = response.text.lstrip('_')
                json_response = json.loads(content)
                return json_response.get('gpt', '')
            except json.JSONDecodeError:
                return "Error: Unable to decode JSON response"
        else:
            return f"Error: {response.status_code}"
