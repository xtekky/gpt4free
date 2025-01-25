from __future__ import annotations

from aiohttp import ClientSession
import time
import asyncio

from ..typing import AsyncResult, Messages
from ..image import ImageResponse
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin


class ImageLabs(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://editor.imagelabs.net"
    api_endpoint = "https://editor.imagelabs.net/txt2img"
    
    working = True
    supports_stream = False
    supports_system_message = False
    supports_message_history = False
    
    default_model = 'general'
    default_image_model = default_model
    image_models = [default_image_model]
    models = image_models
    model_aliases = {"sdxl-turbo": default_model}

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        # Image
        prompt: str = None,
        negative_prompt: str = "",
        width: int = 1152,
        height: int = 896,
        **kwargs
    ) -> AsyncResult:      
        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'no-cache',
            'content-type': 'application/json',
            'origin': cls.url,
            'referer': f'{cls.url}/',
            'x-requested-with': 'XMLHttpRequest',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }
        
        async with ClientSession(headers=headers) as session:
            prompt = messages[-1]["content"] if prompt is None else prompt
            
            # Generate image
            payload = {
                "prompt": prompt,
                "seed": str(int(time.time())),
                "subseed": str(int(time.time() * 1000)),
                "attention": 0,
                "width": width,
                "height": height,
                "tiling": False,
                "negative_prompt": negative_prompt,
                "reference_image": "",
                "reference_image_type": None,
                "reference_strength": 30
            }
            
            async with session.post(f'{cls.url}/txt2img', json=payload, proxy=proxy) as generate_response:
                generate_data = await generate_response.json()
                task_id = generate_data.get('task_id')
            
            # Poll for progress
            while True:
                async with session.post(f'{cls.url}/progress', json={"task_id": task_id}, proxy=proxy) as progress_response:
                    progress_data = await progress_response.json()
                    
                    # Check for completion or error states
                    if progress_data.get('status') == 'Done' or progress_data.get('final_image_url'):
                        # Yield ImageResponse with the final image URL
                        yield ImageResponse(
                            images=[progress_data.get('final_image_url')], 
                            alt=prompt
                        )
                        break
                    
                    # Check for queue or error states
                    if 'error' in progress_data.get('status', '').lower():
                        raise Exception(f"Image generation error: {progress_data}")
                
                # Wait between polls
                await asyncio.sleep(1)

    @classmethod
    def get_model(cls, model: str) -> str:
        return cls.default_model
