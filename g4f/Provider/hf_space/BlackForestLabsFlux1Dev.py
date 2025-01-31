from __future__ import annotations

import json
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ...image import ImageResponse, ImagePreview
from ...errors import ResponseError
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_image_prompt

class BlackForestLabsFlux1Dev(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://black-forest-labs-flux-1-dev.hf.space"
    api_endpoint = "/gradio_api/call/infer"

    working = True

    default_model = 'black-forest-labs-flux-1-dev'
    default_image_model = default_model
    model_aliases = {"flux-dev": default_model, "flux": default_model}
    image_models = [default_image_model, *model_aliases.keys()]
    models = image_models

    @classmethod
    async def create_async_generator(
        cls, 
        model: str, 
        messages: Messages,
        prompt: str = None,
        api_key: str = None, 
        proxy: str = None,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 28,
        seed: int = 0,
        randomize_seed: bool = True,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        async with ClientSession(headers=headers) as session:
            prompt = format_image_prompt(messages, prompt)
            data = {
                "data": [prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps]
            }
            async with session.post(f"{cls.url}{cls.api_endpoint}", json=data, proxy=proxy) as response:
                response.raise_for_status()
                event_id = (await response.json()).get("event_id")
                async with session.get(f"{cls.url}{cls.api_endpoint}/{event_id}") as event_response:
                    event_response.raise_for_status()
                    event = None
                    async for chunk in event_response.content:
                        if chunk.startswith(b"event: "):
                            event = chunk[7:].decode(errors="replace").strip()
                        if chunk.startswith(b"data: "):
                            if event == "error":
                                raise ResponseError(f"GPU token limit exceeded: {chunk.decode(errors='replace')}")
                            if event in ("complete", "generating"):
                                try:
                                    data = json.loads(chunk[6:])
                                    if data is None:
                                        continue
                                    url = data[0]["url"]
                                except (json.JSONDecodeError, KeyError, TypeError) as e:
                                    raise RuntimeError(f"Failed to parse image URL: {chunk.decode(errors='replace')}", e)
                                if event == "generating":
                                    yield ImagePreview(url, prompt)
                                else:
                                    yield ImageResponse(url, prompt)
                                    break
