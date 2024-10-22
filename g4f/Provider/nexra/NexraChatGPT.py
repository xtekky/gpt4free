from __future__ import annotations

import json
import requests

from ...typing import CreateResult, Messages
from ..base_provider import ProviderModelMixin, AbstractProvider
from ..helper import format_prompt

class NexraChatGPT(AbstractProvider, ProviderModelMixin):
    label = "Nexra ChatGPT"
    url = "https://nexra.aryahcr.cc/documentation/chatgpt/en"
    api_endpoint = "https://nexra.aryahcr.cc/api/chat/gpt"
    working = True
    
    default_model = 'gpt-3.5-turbo'
    models = ['gpt-4', 'gpt-4-0613', 'gpt-4-0314', 'gpt-4-32k-0314', default_model, 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-0301', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002', 'gpt-3', 'text-curie-001', 'text-babbage-001', 'text-ada-001', 'davinci', 'curie', 'babbage', 'ada', 'babbage-002', 'davinci-002']
    
    model_aliases = {
        "gpt-4": "gpt-4-0613",
        "gpt-4": "gpt-4-32k",
        "gpt-4": "gpt-4-0314",
        "gpt-4": "gpt-4-32k-0314",
        
        "gpt-3.5-turbo": "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo": "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo": "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo": "gpt-3.5-turbo-0301",
        
        "gpt-3": "text-davinci-003",
        "gpt-3": "text-davinci-002",
        "gpt-3": "code-davinci-002",
        "gpt-3": "text-curie-001",
        "gpt-3": "text-babbage-001",
        "gpt-3": "text-ada-001",
        "gpt-3": "text-ada-001",
        "gpt-3": "davinci",
        "gpt-3": "curie",
        "gpt-3": "babbage",
        "gpt-3": "ada",
        "gpt-3": "babbage-002",
        "gpt-3": "davinci-002",
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
        proxy: str = None,
        markdown: bool = False,
        **kwargs
    ) -> CreateResult:
        model = cls.get_model(model)

        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            "messages": [],
            "prompt": format_prompt(messages),
            "model": model,
            "markdown": markdown
        }
        
        response = requests.post(cls.api_endpoint, headers=headers, json=data)

        return cls.process_response(response)

    @classmethod
    def process_response(cls, response):
        if response.status_code == 200:
            try:
                data = response.json()
                return data.get('gpt', '')
            except json.JSONDecodeError:
                return "Error: Unable to decode JSON response"
        else:
            return f"Error: {response.status_code}"
