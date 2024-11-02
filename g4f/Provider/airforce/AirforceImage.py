from __future__ import annotations

from aiohttp import ClientSession
import random

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...image import ImageResponse


class AirforceImage(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Airforce Image"
    #url = "https://api.airforce"
    api_endpoint_imagine2 = "https://api.airforce/imagine2"
    #working = True
    
    default_model = 'flux'
    image_models = [
        'flux',
        'flux-realism',
        'flux-anime',
        'flux-3d',
        'flux-disney',
        'flux-pixel',
        'flux-4o',
        'any-dark',
        'stable-diffusion-xl-base',
        'stable-diffusion-xl-lightning',
    ]
    models = [*image_models]
    
    model_aliases = {
        "sdxl": "stable-diffusion-xl-base",
        "sdxl": "stable-diffusion-xl-lightning",
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
        size: str = '1:1',
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'authorization': 'Bearer missing api key',
            'cache-control': 'no-cache',
            'origin': 'https://llmplayground.net',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://llmplayground.net/',
            'sec-ch-ua': '"Not?A_Brand";v="99", "Chromium";v="130"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'
        }
        
        async with ClientSession(headers=headers) as session:
            prompt = messages[-1]['content']
            seed = random.randint(0, 4294967295)
            params = {
                'model': model,
                'prompt': prompt,
                'size': size,
                'seed': str(seed)
            }
            async with session.get(cls.api_endpoint_imagine2, params=params, proxy=proxy) as response:
                response.raise_for_status()
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if 'image' in content_type:
                        image_url = str(response.url)
                        yield ImageResponse(image_url, alt="Airforce generated image") 
                    else:
                        content = await response.text()
                        yield f"Unexpected content type: {content_type}\nResponse content: {content}"
                else:
                    error_content = await response.text()
                    yield f"Error: {error_content}"
