from __future__ import annotations

from aiohttp import ClientSession
import time
import random
import asyncio

from ...typing import AsyncResult, Messages
from ...providers.response import ImageResponse, Reasoning, JsonConversation
from ..helper import format_image_prompt, get_random_string
from .DeepseekAI_JanusPro7b import DeepseekAI_JanusPro7b, get_zerogpu_token
from .BlackForestLabs_Flux1Dev import BlackForestLabs_Flux1Dev
from .raise_for_status import raise_for_status

class FluxDev(BlackForestLabs_Flux1Dev):
    url = "https://roxky-flux-1-dev.hf.space"
    space = "roxky/FLUX.1-dev"
    referer = f"{url}/?__theme=light"

class G4F(DeepseekAI_JanusPro7b):
    label = "G4F framework"
    space = "roxky/Janus-Pro-7B"
    url = f"https://huggingface.co/spaces/roxky/g4f-space"
    api_url = "https://roxky-janus-pro-7b.hf.space"
    url_flux = "https://roxky-g4f-flux.hf.space/run/predict"
    referer = f"{api_url}?__theme=light"

    default_model = "flux"
    model_aliases = {"flux-schnell": default_model}
    image_models = [DeepseekAI_JanusPro7b.default_image_model, default_model, "flux-dev", *model_aliases.keys()]
    models = [DeepseekAI_JanusPro7b.default_model, *image_models]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        prompt: str = None,
        aspect_ratio: str = "1:1",
        width: int = None,
        height: int = None,
        seed: int = None,
        cookies: dict = None,
        api_key: str = None,
        zerogpu_uuid: str = "[object Object]",
        **kwargs
    ) -> AsyncResult:
        if model in ("flux", "flux-dev"):
            async for chunk in FluxDev.create_async_generator(
                model, messages,
                proxy=proxy,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                width=width,
                height=height,
                seed=seed,
                cookies=cookies,
                api_key=api_key,
                zerogpu_uuid=zerogpu_uuid,
                **kwargs
            ):
                yield chunk
            return
        if cls.default_model not in model:
            async for chunk in super().create_async_generator(
                model, messages,
                proxy=proxy,
                prompt=prompt,
                seed=seed,
                cookies=cookies, 
                api_key=api_key,
                zerogpu_uuid=zerogpu_uuid,
                **kwargs
            ):
                yield chunk
            return

        model = cls.get_model(model)
        width = max(32, width - (width % 8))
        height = max(32, height - (height % 8))
        if prompt is None:
            prompt = format_image_prompt(messages)
        if seed is None:
            seed = random.randint(9999, 2**32 - 1)

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
            if api_key is None:
                yield Reasoning(status="Acquiring GPU Token")
                zerogpu_uuid, api_key = await get_zerogpu_token(cls.space, session, JsonConversation(), cookies)
            headers = {
                "x-zerogpu-token": api_key,
                "x-zerogpu-uuid": zerogpu_uuid,
            }
            headers = {k: v for k, v in headers.items() if v is not None}
            async def generate():
                async with session.post(cls.url_flux, json=payload, proxy=proxy, headers=headers) as response:
                    await raise_for_status(response)
                    response_data = await response.json()
                    image_url = response_data["data"][0]['url']
                    return ImageResponse(image_url, alt=prompt)
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
