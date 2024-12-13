from __future__ import annotations

import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from ..image import ImageResponse, ImagePreview
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

class Flux(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Flux (HuggingSpace)"
    url = "https://black-forest-labs-flux-1-dev.hf.space"
    api_endpoint = "/gradio_api/call/infer"
    working = True
    default_model = 'flux-dev'
    models = [default_model]
    image_models = [default_model]

    @classmethod
    async def create_async_generator(
        cls, model: str, messages: Messages, prompt: str = None, api_key: str = None, proxy: str = None, **kwargs
    ) -> AsyncResult:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        async with ClientSession(headers=headers) as session:
            prompt = messages[-1]["content"] if prompt is None else prompt
            data = {
                "data": [prompt, 0, True, 1024, 1024, 3.5, 28]
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
                                raise RuntimeError(f"GPU token limit exceeded: {chunk.decode(errors='replace')}")
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
