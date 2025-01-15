from __future__ import annotations

import json
import asyncio
from aiohttp import ClientSession, ContentTypeError

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...requests.aiohttp import get_connector
from ...requests.raise_for_status import raise_for_status
from ..helper import format_prompt
from ...image import ImageResponse

class ReplicateHome(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://replicate.com"
    api_endpoint = "https://homepage.replicate.com/api/prediction"
    
    working = False
    supports_stream = True
    
    default_model = 'google-deepmind/gemma-2b-it'
    default_image_model = 'stability-ai/stable-diffusion-3'
    
    image_models = [
        'stability-ai/stable-diffusion-3',
        'bytedance/sdxl-lightning-4step',
        'playgroundai/playground-v2.5-1024px-aesthetic',
    ]
    
    text_models = [
        'google-deepmind/gemma-2b-it',
    ]

    models = text_models + image_models

    model_aliases = {
        # image_models
        "sd-3": "stability-ai/stable-diffusion-3",
        "sdxl": "bytedance/sdxl-lightning-4step",
        "playground-v2.5": "playgroundai/playground-v2.5-1024px-aesthetic",
        
        # text_models
        "gemma-2b": "google-deepmind/gemma-2b-it",
    }

    model_versions = {
        # image_models
        'stability-ai/stable-diffusion-3': "527d2a6296facb8e47ba1eaf17f142c240c19a30894f437feee9b91cc29d8e4f",
        'bytedance/sdxl-lightning-4step': "5f24084160c9089501c1b3545d9be3c27883ae2239b6f412990e82d4a6210f8f",
        'playgroundai/playground-v2.5-1024px-aesthetic': "a45f82a1382bed5c7aeb861dac7c7d191b0fdf74d8d57c4a0e6ed7d4d0bf7d24",
        
        # text_models
        "google-deepmind/gemma-2b-it": "dff94eaf770e1fc211e425a50b51baa8e4cac6c39ef074681f9e39d778773626",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://replicate.com",
            "referer": "https://replicate.com/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        }
        
        async with ClientSession(headers=headers, connector=get_connector(proxy=proxy)) as session:
            if prompt is None:
                if model in cls.image_models:
                    prompt = messages[-1]['content']
                else:
                    prompt = format_prompt(messages)

            data = {
                "model": model,
                "version": cls.model_versions[model],
                "input": {"prompt": prompt},
            }

            async with session.post(cls.api_endpoint, json=data) as response:
                await raise_for_status(response)
                result = await response.json()
                prediction_id = result['id']

            poll_url = f"https://homepage.replicate.com/api/poll?id={prediction_id}"
            max_attempts = 30
            delay = 5
            for _ in range(max_attempts):
                async with session.get(poll_url) as response:
                    await raise_for_status(response)
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
                            yield ImageResponse(image_url, prompt)
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
