from __future__ import annotations

from aiohttp import ClientSession
import json

from ...typing import AsyncResult, Messages
from ...image import ImageResponse
from ...errors import ResponseError
from ...requests.raise_for_status import raise_for_status
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_image_prompt

class BlackForestLabsFlux1Schnell(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://black-forest-labs-flux-1-schnell.hf.space"
    api_endpoint = "https://black-forest-labs-flux-1-schnell.hf.space/call/infer"
    
    working = True
    
    default_model = "black-forest-labs-flux-1-schnell"
    default_image_model = default_model
    model_aliases = {"flux-schnell": default_model, "flux": default_model}
    image_models = [default_image_model, *model_aliases.keys()]
    models = image_models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        prompt: str = None,
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 2,
        seed: int = 0,
        randomize_seed: bool = True,
        **kwargs
    ) -> AsyncResult:

        model = cls.get_model(model)
        
        width = max(32, width - (width % 8))
        height = max(32, height - (height % 8))

        if prompt is None:
            prompt = format_image_prompt(messages)

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
                await raise_for_status(response)
                response_data = await response.json()
                event_id = response_data['event_id']
                while True:
                    async with session.get(f"{cls.api_endpoint}/{event_id}", proxy=proxy) as status_response:
                        await raise_for_status(status_response)
                        while not status_response.content.at_eof():
                            event = await status_response.content.readuntil(b'\n\n')
                            if event.startswith(b'event:'):
                                event_parts = event.split(b'\ndata: ')
                                if len(event_parts) < 2:
                                    continue
                                event_type = event_parts[0].split(b': ')[1]
                                data = event_parts[1]
                                if event_type == b'error':
                                    raise ResponseError(f"Error generating image: {data}")
                                elif event_type == b'complete':
                                    json_data = json.loads(data)
                                    image_url = json_data[0]['url']
                                    yield ImageResponse(images=[image_url], alt=prompt)
                                    return
