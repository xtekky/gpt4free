from __future__ import annotations

import json
import asyncio
from aiohttp import ClientSession, ContentTypeError

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..image import ImageResponse

class ReplicateHome(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://replicate.com"
    api_endpoint = "https://homepage.replicate.com/api/prediction"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'meta/meta-llama-3-70b-instruct'
    
    text_models = [
        'meta/meta-llama-3-70b-instruct',
        'mistralai/mixtral-8x7b-instruct-v0.1',
        'google-deepmind/gemma-2b-it',
        'yorickvp/llava-13b',
    ]

    image_models = [
        'black-forest-labs/flux-schnell',
        'stability-ai/stable-diffusion-3',
        'bytedance/sdxl-lightning-4step',
        'playgroundai/playground-v2.5-1024px-aesthetic',
    ]

    models = text_models + image_models
    
    model_aliases = {
        "flux-schnell": "black-forest-labs/flux-schnell",
        "sd-3": "stability-ai/stable-diffusion-3",
        "sdxl": "bytedance/sdxl-lightning-4step",
        "playground-v2.5": "playgroundai/playground-v2.5-1024px-aesthetic",
        "llama-3-70b": "meta/meta-llama-3-70b-instruct",
        "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
        "gemma-2b": "google-deepmind/gemma-2b-it",
        "llava-13b": "yorickvp/llava-13b",
    }

    model_versions = {
        "meta/meta-llama-3-70b-instruct": "fbfb20b472b2f3bdd101412a9f70a0ed4fc0ced78a77ff00970ee7a2383c575d",
        "mistralai/mixtral-8x7b-instruct-v0.1": "5d78bcd7a992c4b793465bcdcf551dc2ab9668d12bb7aa714557a21c1e77041c",
        "google-deepmind/gemma-2b-it": "dff94eaf770e1fc211e425a50b51baa8e4cac6c39ef074681f9e39d778773626",
        "yorickvp/llava-13b": "80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb",
        'black-forest-labs/flux-schnell': "f2ab8a5bfe79f02f0789a146cf5e73d2a4ff2684a98c2b303d1e1ff3814271db",
        'stability-ai/stable-diffusion-3': "527d2a6296facb8e47ba1eaf17f142c240c19a30894f437feee9b91cc29d8e4f",
        'bytedance/sdxl-lightning-4step': "5f24084160c9089501c1b3545d9be3c27883ae2239b6f412990e82d4a6210f8f",
        'playgroundai/playground-v2.5-1024px-aesthetic': "a45f82a1382bed5c7aeb861dac7c7d191b0fdf74d8d57c4a0e6ed7d4d0bf7d24",
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
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": "https://replicate.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://replicate.com/",
            "sec-ch-ua": '"Not;A=Brand";v="24", "Chromium";v="128"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        }
        
        async with ClientSession(headers=headers) as session:
            if model in cls.image_models:
                prompt = messages[-1]['content'] if messages else ""
            else:
                prompt = format_prompt(messages)
            
            data = {
                "model": model,
                "version": cls.model_versions[model],
                "input": {"prompt": prompt},
            }
            
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                result = await response.json()
                prediction_id = result['id']
            
            poll_url = f"https://homepage.replicate.com/api/poll?id={prediction_id}"
            max_attempts = 30
            delay = 5
            for _ in range(max_attempts):
                async with session.get(poll_url, proxy=proxy) as response:
                    response.raise_for_status()
                    try:
                        result = await response.json()
                    except ContentTypeError:
                        text = await response.text()
                        try:
                            result = json.loads(text)
                        except json.JSONDecodeError:
                            raise ValueError(f"Unexpected response format: {text}")

                    if result['status'] == 'succeeded':
                        if model in cls.image_models:
                            image_url = result['output'][0]
                            yield ImageResponse(image_url, "Generated image")
                            return
                        else:
                            for chunk in result['output']:
                                yield chunk
                        break
                    elif result['status'] == 'failed':
                        raise Exception(f"Prediction failed: {result.get('error')}")
                await asyncio.sleep(delay)
            
            if result['status'] != 'succeeded':
                raise Exception("Prediction timed out")
