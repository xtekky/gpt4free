from __future__ import annotations

import json
import re
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class TwitterBio(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.twitterbio.io"
    api_endpoint_mistral = "https://www.twitterbio.io/api/mistral"
    api_endpoint_openai = "https://www.twitterbio.io/api/openai"
    working = True
    supports_gpt_35_turbo = True
    
    default_model = 'gpt-3.5-turbo'
    models = [
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'gpt-3.5-turbo',
    ]
    
    model_aliases = {
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        return cls.default_model

    @staticmethod
    def format_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        return text

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
            "origin": cls.url,
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": f"{cls.url}/",
            "sec-ch-ua": '"Chromium";v="127", "Not)A;Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
        }
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "prompt": f'{prompt}.'
            }
            
            if model == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
                api_endpoint = cls.api_endpoint_mistral
            elif model == 'gpt-3.5-turbo':
                api_endpoint = cls.api_endpoint_openai
            else:
                raise ValueError(f"Unsupported model: {model}")
            
            async with session.post(api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                buffer = ""
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            json_data = json.loads(line[6:])
                            if model == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
                                if 'choices' in json_data and len(json_data['choices']) > 0:
                                    text = json_data['choices'][0].get('text', '')
                                    if text:
                                        buffer += text
                            elif model == 'gpt-3.5-turbo':
                                text = json_data.get('text', '')
                                if text:
                                    buffer += text
                        except json.JSONDecodeError:
                            continue
                    elif line == 'data: [DONE]':
                        break
                
                if buffer:
                    yield cls.format_text(buffer)
