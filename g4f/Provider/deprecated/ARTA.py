from __future__ import annotations

import os
import time
import json
import random
from pathlib import Path
from aiohttp import ClientSession, ClientResponse
import asyncio

from ...typing import AsyncResult, Messages
from ...providers.response import ImageResponse, Reasoning
from ...errors import ResponseError, ModelNotFoundError
from ...cookies import get_cookies_dir
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_media_prompt
from ... import debug

class ARTA(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://ai-arta.com"
    auth_url = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/signupNewUser?key=AIzaSyB3-71wG0fIt0shj0ee4fvx1shcjJHGrrQ"
    token_refresh_url = "https://securetoken.googleapis.com/v1/token?key=AIzaSyB3-71wG0fIt0shj0ee4fvx1shcjJHGrrQ"
    image_generation_url = "https://img-gen-prod.ai-arta.com/api/v1/text2image"
    status_check_url = "https://img-gen-prod.ai-arta.com/api/v1/text2image/{record_id}/status"

    working = False # Take down request

    default_model = "flux"
    default_image_model = default_model
    model_aliases = {
        "anything-xl": "Anything-xl",
        "high-gpt4o": "High GPT4o",
        "on-limbs-black": "On limbs black",
        "f-dev": "F Dev",
        "flux-dev": "F Dev", # Added
        "sdxl-1.0": "SDXL 1.0", # Added
        "old-school": "Old School",
        "vincent-van-gogh": "Vincent Van Gogh",
        "cor-epica-xl": "Cor-epica-xl",
        "professional": "Professional",
        "cheyenne-xl": "Cheyenne-xl",
        "chicano": "Chicano",
        "sdxl-l": "SDXL L", # Added
        "black-ink": "Black Ink",
        "juggernaut-xl": "Juggernaut-xl",
        "cinematic-art": "Cinematic Art",
        "dreamshaper-xl": "Dreamshaper-xl",
        "fantasy-art": "Fantasy Art",
        "neo-traditional": "Neo-traditional",
        "realistic-stock-xl": "Realistic-stock-xl",
        "flame-design": "Flame design",
        "japanese-2": "Japanese_2",
        "medieval": "Medieval",
        "surrealism": "Surrealism",
        "dotwork": "Dotwork",
        "graffiti": "Graffiti",
        "revanimated": "RevAnimated",
        "on-limbs-color": "On limbs color",
        "old-school-colored": "Old school colored",
        "gpt4o-ghibli": "GPT4o Ghibli",
        "low-poly": "Low Poly",
        "gpt4o": "GPT4o",
        "gpt-image": ["GPT4o", "High GPT4o", "GPT4o Ghibli"],
        "no-style": "No Style",
        "anime": "Anime",
        "tattoo": "tattoo",
        "embroidery-tattoo": "Embroidery tattoo",
        "mini-tattoo": "Mini tattoo",
        "realistic-tattoo": "Realistic tattoo",
        "playground-xl": "Playground-xl",
        "Watercolor": "Watercolor",
        "f-pro": "F Pro",
        "flux-pro": "F Pro", # Added
        "kawaii": "Kawaii",
        "photographic": "Photographic",
        "katayama-mix-xl": "Katayama-mix-xl",
        "death-metal": "Death metal",
        "new-school": "New School",
        "pony-xl": "Pony-xl",
        "anima-pencil-xl": "Anima-pencil-xl",
        default_image_model: "Flux", # Added
        "biomech": "Biomech",
        "yamers-realistic-xl": "Yamers-realistic-xl",
        "trash-polka": "Trash Polka",
        "red-and-black": "Red and Black",
    }
    image_models = list(model_aliases.keys())
    models = image_models

    @classmethod
    def get_model(cls, model: str) -> str:
        """Get the internal model name from the user-provided model name."""
        if not model:
            return cls.model_aliases[cls.default_model]
        
        # Always check aliases first to get the proper API name
        if model in cls.model_aliases:
            alias = cls.model_aliases[model]
            # If the alias is a list, randomly select one of the options
            if isinstance(alias, list):
                selected_model = random.choice(alias)
                debug.log(f"ARTA: Selected model '{selected_model}' from alias '{model}'")
                return selected_model
            debug.log(f"ARTA: Using model '{alias}' for alias '{model}'")
            return alias
        
        # If not in aliases, check if it's a direct API model name
        api_model_names = [v for v in cls.model_aliases.values() if isinstance(v, str)]
        if model in api_model_names:
            return model
        
        raise ModelNotFoundError(f"Model {model} not found")



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
                await raise_error(f"Failed to obtain authentication token", auth_response)
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
                await raise_error(f"Failed to refresh token", response)
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
        aspect_ratio: str = None,
        seed: int = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        prompt = format_media_prompt(messages, prompt)

        # Generate a random seed if not provided
        if seed is None:
            seed = random.randint(9999, 99999999)  # Common range for random seeds

        # Step 1: Get Authentication Token
        auth_data = await cls.read_and_refresh_token(proxy)
        auth_token = auth_data.get("idToken")

        async with ClientSession() as session:
            # Step 2: Generate Images
            # Create a form data structure as the API might expect form data instead of JSON
            form_data = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "style": model,
                "images_num": str(n),
                "cfg_scale": str(guidance_scale),
                "steps": str(num_inference_steps),
                "aspect_ratio": "1:1" if aspect_ratio is None else aspect_ratio,
                "seed": str(seed),
            }

            headers = {
                "Authorization": auth_token,
                # No Content-Type header for multipart/form-data, aiohttp sets it automatically
            }

            # Try with form data instead of JSON
            async with session.post(cls.image_generation_url, data=form_data, headers=headers, proxy=proxy) as image_response:
                await raise_error(f"Failed to initiate image generation", image_response)
                image_data = await image_response.json()
                record_id = image_data.get("record_id")
                if not record_id:
                    raise ResponseError(f"Failed to initiate image generation: {image_data}")

            # Step 3: Check Generation Status
            status_url = cls.status_check_url.format(record_id=record_id)
            start_time = time.time()
            last_status = None
            while True:
                async with session.get(status_url, headers=headers, proxy=proxy) as status_response:
                    await raise_error(f"Failed to check image generation status", status_response)
                    status_data = await status_response.json()
                    status = status_data.get("status")

                    if status == "DONE":
                        image_urls = [image["url"] for image in status_data.get("response", [])]
                        duration = time.time() - start_time
                        yield Reasoning(label="Generated", status=f"{n} image in {duration:.2f}s" if n == 1 else f"{n} images in {duration:.2f}s")
                        yield ImageResponse(urls=image_urls, alt=prompt)
                        return
                    elif status in ("IN_QUEUE", "IN_PROGRESS"):
                        if last_status != status:
                            last_status = status
                            if status == "IN_QUEUE":
                                yield Reasoning(label="Waiting")
                            else:
                                yield Reasoning(label="Generating")
                        await asyncio.sleep(2)  # Poll every 2 seconds
                    else:
                        raise ResponseError(f"Image generation failed with status: {status}")

async def raise_error(message: str, response: ClientResponse):
    if response.ok:
        return
    error_text = await response.text()
    content_type = response.headers.get('Content-Type', 'unknown')
    raise ResponseError(f"{message}. Content-Type: {content_type}, Response: {error_text}")
