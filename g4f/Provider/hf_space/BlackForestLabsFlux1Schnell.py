from __future__ import annotations

from aiohttp import ClientSession
import json
import random
from typing import Optional

from ...typing import AsyncResult, Messages
from ...image import ImageResponse
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin

class BlackForestLabsFlux1Schnell(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://black-forest-labs-flux-1-schnell.hf.space"
    api_endpoint = "https://black-forest-labs-flux-1-schnell.hf.space/call/infer"
    
    working = True
    
    default_model = "flux-schnell"
    default_image_model = default_model
    image_models = [default_image_model]
    models = [*image_models]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 2,
        seed: Optional[int] = None,
        randomize_seed: bool = False,
        **kwargs
    ) -> AsyncResult:
        if seed is None:
            seed = random.randint(0, 10000)
        
        width = max(32, width - (width % 8))
        height = max(32, height - (height % 8))
        
        prompt = messages[-1]["content"]
        
        payload = {
            "data": [
                prompt,
                seed,
                randomize_seed,
                width,
                height,
                num_inference_steps
            ]
        }

        async with ClientSession() as session:
            async with session.post(cls.api_endpoint, json=payload, proxy=proxy) as response:
                response.raise_for_status()
                response_data = await response.json()
                event_id = response_data['event_id']

                while True:
                    async with session.get(f"{cls.api_endpoint}/{event_id}", proxy=proxy) as status_response:
                        status_response.raise_for_status()
                        events = (await status_response.text()).split('\n\n')
                        
                        for event in events:
                            if event.startswith('event:'):
                                event_parts = event.split('\ndata: ')
                                if len(event_parts) < 2:
                                    continue
                                    
                                event_type = event_parts[0].split(': ')[1]
                                data = event_parts[1]

                                if event_type == 'error':
                                    raise Exception(f"Error generating image: {data}")
                                elif event_type == 'complete':
                                    json_data = json.loads(data)
                                    image_url = json_data[0]['url']
                                    yield ImageResponse(images=[image_url], alt=prompt)
                                    return
