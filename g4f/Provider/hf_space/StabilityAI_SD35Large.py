from __future__ import annotations

import json
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ...providers.response import ImageResponse, ImagePreview
from ...image import use_aspect_ratio
from ...errors import ResponseError
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_media_prompt

class StabilityAI_SD35Large(AsyncGeneratorProvider, ProviderModelMixin):
    label = "StabilityAI SD-3.5-Large"
    url = "https://stabilityai-stable-diffusion-3-5-large.hf.space"
    api_endpoint = "/gradio_api/call/infer"

    working = True

    default_model = 'stabilityai-stable-diffusion-3-5-large'
    default_image_model = default_model
    model_aliases = {"sd-3.5-large": default_model}
    image_models = list(model_aliases.keys())
    models = image_models

    @classmethod
    async def create_async_generator(
        cls, model: str, messages: Messages,
        prompt: str = None,
        negative_prompt: str = None,
        api_key: str = None, 
        proxy: str = None,
        aspect_ratio: str = "1:1",
        width: int = None,
        height: int = None,
        guidance_scale: float = 4.5,
        num_inference_steps: int = 50,
        seed: int = 0,
        randomize_seed: bool = True,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        async with ClientSession(headers=headers) as session:
            prompt = format_media_prompt(messages, prompt)
            data = use_aspect_ratio({"width": width, "height": height}, aspect_ratio)
            data = {
                "data": [prompt, negative_prompt, seed, randomize_seed, data.get("width"), data.get("height"), guidance_scale, num_inference_steps]
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
