from __future__ import annotations

import os
import time
import json
import random
from pathlib import Path
from aiohttp import ClientSession
import asyncio

from ..typing import AsyncResult, Messages
from ..providers.response import ImageResponse, Reasoning
from ..errors import ResponseError
from ..cookies import get_cookies_dir
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_image_prompt

class ARTA(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://img-gen-prod.ai-arta.com"
    auth_url = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/signupNewUser?key=AIzaSyB3-71wG0fIt0shj0ee4fvx1shcjJHGrrQ"
    token_refresh_url = "https://securetoken.googleapis.com/v1/token?key=AIzaSyB3-71wG0fIt0shj0ee4fvx1shcjJHGrrQ"
    image_generation_url = "https://img-gen-prod.ai-arta.com/api/v1/text2image"
    status_check_url = "https://img-gen-prod.ai-arta.com/api/v1/text2image/{record_id}/status"

    working = True

    default_model = "Flux"
    default_image_model = default_model
    model_aliases = {
        "flux": "Flux",
        "medieval": "Medieval",
        "vincent_van_gogh": "Vincent Van Gogh",
        "f_dev": "F Dev",
        "low_poly": "Low Poly",
        "dreamshaper_xl": "Dreamshaper-xl",
        "anima_pencil_xl": "Anima-pencil-xl",
        "biomech": "Biomech",
        "trash_polka": "Trash Polka",
        "no_style": "No Style",
        "cheyenne_xl": "Cheyenne-xl",
        "chicano": "Chicano",
        "embroidery_tattoo": "Embroidery tattoo",
        "red_and_black": "Red and Black",
        "fantasy_art": "Fantasy Art",
        "watercolor": "Watercolor",
        "dotwork": "Dotwork",
        "old_school_colored": "Old school colored",
        "realistic_tattoo": "Realistic tattoo",
        "japanese_2": "Japanese_2",
        "realistic_stock_xl": "Realistic-stock-xl",
        "f_pro": "F Pro",
        "revanimated": "RevAnimated",
        "katayama_mix_xl": "Katayama-mix-xl",
        "sdxl_l": "SDXL L",
        "cor_epica_xl": "Cor-epica-xl",
        "anime_tattoo": "Anime tattoo",
        "new_school": "New School",
        "death_metal": "Death metal",
        "old_school": "Old School",
        "juggernaut_xl": "Juggernaut-xl",
        "photographic": "Photographic",
        "sdxl_1_0": "SDXL 1.0",
        "graffiti": "Graffiti",
        "mini_tattoo": "Mini tattoo",
        "surrealism": "Surrealism",
        "neo_traditional": "Neo-traditional",
        "on_limbs_black": "On limbs black",
        "yamers_realistic_xl": "Yamers-realistic-xl",
        "pony_xl": "Pony-xl",
        "playground_xl": "Playground-xl",
        "anything_xl": "Anything-xl",
        "flame_design": "Flame design",
        "kawaii": "Kawaii",
        "cinematic_art": "Cinematic Art",
        "professional": "Professional",
        "black_ink": "Black Ink"
    }
    image_models = [*model_aliases.keys()]
    models = image_models

    @classmethod
    def get_auth_file(cls):
        path = Path(get_cookies_dir())
        path.mkdir(exist_ok=True)
        filename = f"auth_{cls.__name__}.json"
        return path / filename

    @classmethod
    async def create_token(cls, path: Path, proxy: str | None = None):
        async with ClientSession() as session:
            # Step 1: Generate Authentication Token
            auth_payload = {"clientType": "CLIENT_TYPE_ANDROID"}
            async with session.post(cls.auth_url, json=auth_payload, proxy=proxy) as auth_response:
                auth_data = await auth_response.json()
                auth_token = auth_data.get("idToken")
                #refresh_token = auth_data.get("refreshToken")
                if not auth_token:
                    raise ResponseError("Failed to obtain authentication token.")
                json.dump(auth_data, path.open("w"))
                return auth_data

    @classmethod
    async def refresh_token(cls, refresh_token: str, proxy: str = None) -> tuple[str, str]:
        async with ClientSession() as session:
            payload = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }
            async with session.post(cls.token_refresh_url, data=payload, proxy=proxy) as response:
                response_data = await response.json()
                return response_data.get("id_token"), response_data.get("refresh_token")

    @classmethod
    async def read_and_refresh_token(cls, proxy: str | None = None) -> str:
        path = cls.get_auth_file()
        if path.is_file():
            auth_data = json.load(path.open("rb"))
            diff = time.time() - os.path.getmtime(path)
            expiresIn = int(auth_data.get("expiresIn"))
            if diff < expiresIn:
                if diff > expiresIn / 2:
                    auth_data["idToken"], auth_data["refreshToken"] = await cls.refresh_token(auth_data.get("refreshToken"), proxy)
                    json.dump(auth_data, path.open("w"))
                return auth_data
        return await cls.create_token(path, proxy)

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        prompt: str = None,
        negative_prompt: str = "blurry, deformed hands, ugly",
        n: int = 1,
        guidance_scale: int = 7,
        num_inference_steps: int = 30,
        aspect_ratio: str = "1:1",
        seed: int = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        prompt = format_image_prompt(messages, prompt)

        # Generate a random seed if not provided
        if seed is None:
            seed = random.randint(9999, 99999999)  # Common range for random seeds

        # Step 1: Get Authentication Token
        auth_data = await cls.read_and_refresh_token(proxy)

        async with ClientSession() as session:
            # Step 2: Generate Images
            image_payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "style": model,
                "images_num": str(n),
                "cfg_scale": str(guidance_scale),
                "steps": str(num_inference_steps),
                "aspect_ratio": aspect_ratio,
                "seed": str(seed),
            }

            headers = {
                "Authorization": auth_data.get("idToken"),
            }

            async with session.post(cls.image_generation_url, data=image_payload, headers=headers, proxy=proxy) as image_response:
                image_data = await image_response.json()
                record_id = image_data.get("record_id")

                if not record_id:
                    raise ResponseError(f"Failed to initiate image generation: {image_data}")

            # Step 3: Check Generation Status
            status_url = cls.status_check_url.format(record_id=record_id)
            counter = 0
            while True:
                async with session.get(status_url, headers=headers, proxy=proxy) as status_response:
                    status_data = await status_response.json()
                    status = status_data.get("status")

                    if status == "DONE":
                        image_urls = [image["url"] for image in status_data.get("response", [])]
                        yield Reasoning(status="Finished")
                        yield ImageResponse(images=image_urls, alt=prompt)
                        return
                    elif status in ("IN_QUEUE", "IN_PROGRESS"):
                        yield Reasoning(status=("Waiting" if status == "IN_QUEUE" else "Generating") + "." * counter)
                        await asyncio.sleep(2)  # Poll every 5 seconds
                        counter += 1
                        if counter > 3:
                            counter = 0
                    else:
                        raise ResponseError(f"Image generation failed with status: {status}")