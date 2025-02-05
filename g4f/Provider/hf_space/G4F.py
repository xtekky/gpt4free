from __future__ import annotations

from aiohttp import ClientSession
import time
import asyncio

from ...typing import AsyncResult, Messages
from ...providers.response import ImageResponse, Reasoning
from ...requests.raise_for_status import raise_for_status
from ..helper import format_image_prompt, get_random_string
from .Janus_Pro_7B import Janus_Pro_7B, JsonConversation, get_zerogpu_token

class G4F(Janus_Pro_7B):
    label = "G4F framework"
    space = "roxky/Janus-Pro-7B"
    url = f"https://huggingface.co/spaces/roxky/g4f-space"
    api_url = "https://roxky-janus-pro-7b.hf.space"
    url_flux = "https://roxky-g4f-flux.hf.space/run/predict"
    referer = f"{api_url}?__theme=light"

    default_model = "flux"
    model_aliases = {"flux-schnell": default_model, "flux-dev": default_model}
    image_models = [Janus_Pro_7B.default_image_model, default_model, *model_aliases.keys()]
    models = [Janus_Pro_7B.default_model, *image_models]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        prompt: str = None,
        width: int = 1024,
        height: int = 1024,
        seed: int = None,
        cookies: dict = None,
        **kwargs
    ) -> AsyncResult:
        if cls.default_model not in model:
            async for chunk in super().create_async_generator(model, messages, prompt=prompt, seed=seed, cookies=cookies, **kwargs):
                yield chunk
            return

        model = cls.get_model(model)
        width = max(32, width - (width % 8))
        height = max(32, height - (height % 8))
        if prompt is None:
            prompt = format_image_prompt(messages)
        if seed is None:
            seed = int(time.time())

        payload = {
            "data": [
                prompt,
                seed,
                width,
                height,
                True,
                1
            ],
            "event_data": None,
            "fn_index": 3,
            "session_hash": get_random_string(),
            "trigger_id": 10
        }
        async with ClientSession() as session:
            yield Reasoning(status="Acquiring GPU Token")
            zerogpu_uuid, zerogpu_token = await get_zerogpu_token(cls.space, session, JsonConversation(), cookies)
            headers = {
                "x-zerogpu-token": zerogpu_token,
                "x-zerogpu-uuid": zerogpu_uuid,
            }
            async def generate():
                async with session.post(cls.url_flux, json=payload, proxy=proxy, headers=headers) as response:
                    await raise_for_status(response)
                    response_data = await response.json()
                    image_url = response_data["data"][0]['url']
                    return ImageResponse(images=[image_url], alt=prompt)
            background_tasks = set()
            started = time.time()
            task = asyncio.create_task(generate())
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)
            while background_tasks:
                yield Reasoning(status=f"Generating {time.time() - started:.2f}s")
                await asyncio.sleep(0.2)
            yield await task
            yield Reasoning(status=f"Finished {time.time() - started:.2f}s")