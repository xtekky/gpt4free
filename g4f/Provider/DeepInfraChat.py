from __future__ import annotations

from aiohttp import ClientSession
import json

from ..typing import AsyncResult, Messages, ImageType
from ..image import to_data_uri
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class DeepInfraChat(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://deepinfra.com/chat"
    api_endpoint = "https://api.deepinfra.com/v1/openai/chat/completions"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
    models = [
        'meta-llama/Meta-Llama-3.1-405B-Instruct',
        'meta-llama/Meta-Llama-3.1-70B-Instruct',
        'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'mistralai/Mixtral-8x22B-Instruct-v0.1',
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'microsoft/WizardLM-2-8x22B',
        'microsoft/WizardLM-2-7B',
        'Qwen/Qwen2-72B-Instruct',
        'microsoft/Phi-3-medium-4k-instruct',
        'google/gemma-2-27b-it',
        'openbmb/MiniCPM-Llama3-V-2_5', # Image upload is available
        'mistralai/Mistral-7B-Instruct-v0.3',
        'lizpreciatior/lzlv_70b_fp16_hf',
        'openchat/openchat-3.6-8b',
        'Phind/Phind-CodeLlama-34B-v2',
        'cognitivecomputations/dolphin-2.9.1-llama-3-70b',
    ]
    model_aliases = {
        "llama-3.1-405b": "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "wizardlm-2-8x22b": "microsoft/WizardLM-2-8x22B",
        "wizardlm-2-7b": "microsoft/WizardLM-2-7B",
        "qwen-2-72b": "Qwen/Qwen2-72B-Instruct",
        "phi-3-medium-4k": "microsoft/Phi-3-medium-4k-instruct",
        "gemma-2b-27b": "google/gemma-2-27b-it",
        "minicpm-llama-3-v2.5": "openbmb/MiniCPM-Llama3-V-2_5", # Image upload is available
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "lzlv-70b": "lizpreciatior/lzlv_70b_fp16_hf",
        "openchat-3.6-8b": "openchat/openchat-3.6-8b",
        "phind-codellama-34b-v2": "Phind/Phind-CodeLlama-34B-v2",
        "dolphin-2.9.1-llama-3-70b": "cognitivecomputations/dolphin-2.9.1-llama-3-70b",
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
        image: ImageType = None,
        image_name: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'Origin': 'https://deepinfra.com',
            'Pragma': 'no-cache',
            'Referer': 'https://deepinfra.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'X-Deepinfra-Source': 'web-embed',
            'accept': 'text/event-stream',
            'sec-ch-ua': '"Not;A=Brand";v="24", "Chromium";v="128"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
        }
        
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': 'Be a helpful assistant'},
                    {'role': 'user', 'content': prompt}
                ],
                'stream': True
            }

            if model == 'openbmb/MiniCPM-Llama3-V-2_5' and image is not None:
                data['messages'][-1]['content'] = [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': to_data_uri(image)
                        }
                    },
                    {
                        'type': 'text',
                        'text': messages[-1]['content']
                    }
                ]

            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith('data:'):
                            json_part = decoded_line[5:].strip()
                            if json_part == '[DONE]':
                                break
                            try:
                                data = json.loads(json_part)
                                choices = data.get('choices', [])
                                if choices:
                                    delta = choices[0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                print(f"JSON decode error: {json_part}")
