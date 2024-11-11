from __future__ import annotations

from aiohttp import ClientSession
from urllib.parse import urlencode
import random
import requests

from ...typing import AsyncResult, Messages
from ...image import ImageResponse
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin


class AirforceImage(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Airforce Image"
    #url = "https://api.airforce"
    api_endpoint = "https://api.airforce/imagine2"
    #working = True
    
    default_model = 'flux'
    
    response = requests.get('https://api.airforce/imagine/models')
    data = response.json()

    image_models = data

    models = [*image_models, "stable-diffusion-xl-base", "stable-diffusion-xl-lightning", "Flux-1.1-Pro"]
    
    model_aliases = {
        "sdxl": "stable-diffusion-xl-base",
        "sdxl": "stable-diffusion-xl-lightning", 
        "flux-pro": "Flux-1.1-Pro",
    }
    
    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        else:
            return cls.default_model

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        size: str = '1:1', # "1:1", "16:9", "9:16", "21:9", "9:21", "1:2", "2:1"
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            'accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'no-cache',
            'dnt': '1',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://llmplayground.net/',
            'sec-ch-ua': '"Not?A_Brand";v="99", "Chromium";v="130"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
            'sec-fetch-dest': 'image',
            'sec-fetch-mode': 'no-cors', 
            'sec-fetch-site': 'cross-site',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'
        }

        async with ClientSession(headers=headers) as session:
            seed = random.randint(0, 58463)
            params = {
                'model': model,
                'prompt': messages[-1]["content"],
                'size': size,
                'seed': seed
            }
            full_url = f"{cls.api_endpoint}?{urlencode(params)}"

            async with session.get(full_url, headers=headers, proxy=proxy) as response:
                if response.status == 200 and response.headers.get('content-type', '').startswith('image'):
                    yield ImageResponse(images=[full_url], alt="Generated Image")
                else:
                    raise Exception(f"Error: status {response.status}, content type {response.headers.get('content-type')}")
