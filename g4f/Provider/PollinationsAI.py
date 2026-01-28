from __future__ import annotations

import time
import random
import requests
import asyncio
import json
from urllib.parse import quote, quote_plus
from datetime import datetime
from typing import Optional
from aiohttp import ClientSession, ClientTimeout
from pathlib import Path

from .helper import filter_none, format_media_prompt
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..typing import AsyncResult, Messages, MediaListType
from ..image import is_data_an_audio
from ..errors import MissingAuthError
from ..requests.defaults import DEFAULT_HEADERS
from ..requests.raise_for_status import raise_for_status
from ..requests.aiohttp import get_connector
from ..image import use_aspect_ratio
from ..providers.response import ImageResponse, Reasoning, VideoResponse, JsonRequest, PreviewResponse
from ..tools.media import render_messages
from ..tools.run_tools import AuthManager
from ..cookies import get_cookies_dir
from ..tools.files import secure_filename
from .template.OpenaiTemplate import read_response
from .. import debug

class PollinationsAI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Pollinations AI 🌸"
    url = "https://pollinations.ai"
    login_url = "https://enter.pollinations.ai"
    active_by_default = True
    working = True
    supports_system_message = True
    supports_message_history = True

    # API endpoints
    text_api_endpoint = "https://text.pollinations.ai/openai"
    image_api_endpoint = "https://image.pollinations.ai/prompt/{}"
    gen_image_api_endpoint = "https://gen.pollinations.ai/image/{}"
    gen_text_api_endpoint = "https://gen.pollinations.ai/v1/chat/completions"
    image_models_endpoint = "https://gen.pollinations.ai/image/models"
    text_models_endpoint = "https://gen.pollinations.ai/text/models"
    balance_endpoint = "https://g4f.space/api/pollinations/account/balance"
    worker_api_endpoint = "https://g4f.space/api/pollinations/chat/completions"
    worker_models_endpoint = "https://g4f.space/api/pollinations/models"

    # Models configuration
    default_model = "openai"
    fallback_model = "deepseek"
    default_image_model = "flux"
    default_vision_model = default_model
    default_voice = "alloy"
    text_models = [default_model]
    image_models = [default_image_model, "turbo", "kontext"]
    audio_models = {}
    vision_models = [default_vision_model]
    model_aliases = {
        "gpt-4.1-nano": "openai-fast",
        "llama-4-scout": "llamascout",
        "deepseek-r1": "deepseek-reasoning",
        "mistral-small-3.1-24b": "mistral-small",
        "qwen-2.5-coder-32b": "qwen-3-coder",
        "sdxl-turbo": "turbo",
        "gpt-image": "gptimage",
        "flux-dev": "flux",
        "flux-schnell": "flux",
        "flux-pro": "flux",
        "flux": "flux",
        "flux-kontext": "kontext",
    }
    swap_model_aliases = {v: k for k, v in model_aliases.items()}
    balance: Optional[float] = None
    current_models_endpoint: Optional[str] = None

    @classmethod
    def get_balance(cls, api_key: str, timeout: Optional[float] = None) -> Optional[float]:
        try:
            headers = None
            if api_key:
                headers = {"authorization": f"Bearer {api_key}"}
            response = requests.get(cls.balance_endpoint, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            cls.balance = float(data.get("balance", 0.0))
            debug.log(f"Pollinations AI balance: {cls.balance:.2f} Pollen")
            return cls.balance
        except Exception as e:
            debug.error(f"Failed to get balance:", e)
            return None

    @classmethod
    def get_models(cls, api_key: Optional[str] = None, timeout: Optional[float] = None, **kwargs):
        def get_alias(model: dict) -> str:
            alias = model.get("name")
            if (model.get("aliases")):
                alias = model.get("aliases")[0]
            elif alias in cls.swap_model_aliases:
                alias = cls.swap_model_aliases[alias]
            if alias == "searchgpt":
                return model.get("name")
            return str(alias).replace("-instruct", "").replace("qwen-", "qwen").replace("qwen", "qwen-")
        
        if not api_key:
            api_key = AuthManager.load_api_key(cls)
        if (not api_key or api_key.startswith("g4f_") or api_key.startswith("gfs_")) and cls.balance or cls.balance is None and cls.get_balance(api_key, timeout) and cls.balance > 0:
            debug.log(f"Authenticated with Pollinations AI using G4F API.")
            models_url = cls.worker_models_endpoint
        elif api_key:
            debug.log(f"Using Pollinations AI with provided API key.")
            models_url = cls.gen_text_api_endpoint
        else:
            debug.log(f"Using Pollinations AI without authentication.")
            models_url = cls.text_models_endpoint

        if cls.current_models_endpoint != models_url:
            path = Path(get_cookies_dir()) / "models" / datetime.today().strftime('%Y-%m-%d') / f"{secure_filename(models_url)}.json"
            if path.exists():
                try:
                    data = path.read_text()
                    models_data = json.loads(data)
                    for key, value in models_data.items():
                        setattr(cls, key, value)
                    return cls.models
                except Exception as e:
                    debug.error(f"Failed to load cached models from {path}: {e}")
            try:
                # Update of image models
                image_response = requests.get(cls.image_models_endpoint, timeout=timeout)
                if image_response.ok:
                    new_image_models = image_response.json()
                else:
                    new_image_models = []

                # Combine image models without duplicates
                image_models = cls.image_models.copy()  # Start with default model

                # Add extra image models if not already in the list
                for model in new_image_models:
                    alias = get_alias(model) if isinstance(model, dict) else model
                    if model not in image_models:
                        if isinstance(model, str) or "image" in model.get("output_modalities", []):
                            image_models.append(alias)
                    if isinstance(model, dict) and alias != model.get("name"):
                        cls.model_aliases[alias] = model.get("name")

                cls.image_models = image_models
                cls.video_models = [get_alias(model) for model in new_image_models if isinstance(model, dict) and "video" in model.get("output_modalities", [])]

                text_response = requests.get(cls.text_models_endpoint, timeout=timeout)
                if not text_response.ok:
                    text_response = requests.get(cls.text_models_endpoint, timeout=timeout)
                text_response.raise_for_status()
                models = text_response.json()

                # Purpose of audio models
                cls.audio_models = {
                    model.get("name"): model.get("voices")
                    for model in models
                    if "output_modalities" in model and "audio" in model["output_modalities"]
                }
                for alias, model in cls.model_aliases.items():
                    if model in cls.audio_models and alias not in cls.audio_models:
                        cls.audio_models.update({alias: {}})

                cls.vision_models.extend([
                    get_alias(model)
                    for model in models
                    if model.get("vision") and get_alias(model) not in cls.vision_models
                ])

                for model in models:
                    alias = get_alias(model)
                    if alias != model.get("name"):
                        cls.model_aliases[alias] = model.get("name")
                    if alias not in cls.text_models:
                        cls.text_models.append(alias)
                    elif model.get("name") not in cls.text_models:
                        cls.text_models.append(model.get("name"))
                cls.live += 1
                cls.swap_model_aliases = {v: k for k, v in cls.model_aliases.items()}

            finally:
                cls.current_models_endpoint = models_url
            # Return unique models across all categories
            all_models = cls.text_models.copy()
            all_models.extend(cls.image_models)
            all_models.extend(cls.audio_models.keys())
            cls.models = all_models
            # Cache the models to a file
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    json.dump({
                        "text_models": cls.text_models,
                        "image_models": cls.image_models,
                        "video_models": cls.video_models,
                        "audio_models": cls.audio_models,
                        "vision_models": cls.vision_models,
                        "model_aliases": cls.model_aliases,
                        "models": cls.models,
                        "swap_model_aliases": cls.swap_model_aliases,
                    }, f, indent=4)
            except Exception as e:
                debug.error(f"Failed to cache models to {path}: {e}")
        return cls.models

    @classmethod
    def get_grouped_models(cls, **kwargs) -> dict[str, list[str]]:
        cls.get_models(**kwargs)
        return [
            {"group": "Text Generation", "models": cls.text_models},
            {"group": "Image Generation", "models": cls.image_models},
            {"group": "Video Generation", "models": cls.video_models},
            {"group": "Audio Generation", "models": list(cls.audio_models.keys())},
        ]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        proxy: str = None,
        cache: bool = None,
        api_key: str = None,
        extra_body: dict = None,
        # Image generation parameters
        prompt: str = None,
        aspect_ratio: str = None,
        width: int = None,
        height: int = None,
        seed: Optional[int] = None,
        nologo: bool = True,
        private: bool = False,
        enhance: bool = None,
        safe: bool = False,
        transparent: bool = False,
        n: int = 1,
        # Text generation parameters
        media: MediaListType = None,
        temperature: float = None,
        presence_penalty: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        response_format: Optional[dict] = None,
        extra_parameters: list[str] = ["tools", "parallel_tool_calls", "tool_choice", "reasoning_effort",
                                        "logit_bias", "voice", "modalities", "audio"],
        **kwargs
    ) -> AsyncResult:
        if cache is None:
            cache = kwargs.get("action") is None or kwargs.get("action") != "variant"
        if extra_body is None:
            extra_body = {}
        if not model:
            has_audio = "audio" in kwargs or "audio" in kwargs.get("modalities", [])
            if not has_audio and media is not None:
                for media_data, filename in media:
                    if is_data_an_audio(media_data, filename):
                        has_audio = True
                        break
            model = "openai-audio" if has_audio else cls.default_model
        if not api_key:
            api_key = AuthManager.load_api_key(cls)
        if cls.get_models(api_key=api_key, timeout=kwargs.get("timeout", 15)):
            if model in cls.model_aliases:
                model = cls.model_aliases[model]
        debug.log(f"Using model: {model}")
        alias = cls.swap_model_aliases.get(model, model)
        if alias in cls.image_models or alias in cls.video_models:
            async for chunk in cls._generate_image(
                model="gptimage" if model == "transparent" else model,
                alias=alias,
                prompt=format_media_prompt(messages, prompt),
                media=media,
                proxy=proxy,
                aspect_ratio=aspect_ratio,
                width=width,
                height=height,
                seed=seed,
                cache=cache,
                nologo=nologo,
                private=private,
                enhance=enhance,
                safe=safe,
                transparent=transparent or model == "transparent",
                n=n,
                api_key=api_key
            ):
                yield chunk
        else:
            if prompt is not None and len(messages) == 1:
                messages = [{
                    "role": "user",
                    "content": prompt
                }]
            async for result in cls._generate_text(
                    model=model,
                    messages=messages,
                    media=media,
                    proxy=proxy,
                    temperature=temperature,
                    presence_penalty=presence_penalty,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    response_format=response_format,
                    seed=seed,
                    cache=cache,
                    stream=stream,
                    extra_parameters=extra_parameters,
                    api_key=api_key,
                    extra_body=extra_body,
                    **kwargs
            ):
                yield result

    @classmethod
    async def _generate_image(
        cls,
        model: str,
        alias: str,
        prompt: str,
        media: MediaListType,
        proxy: str,
        aspect_ratio: str,
        width: int,
        height: int,
        seed: Optional[int],
        cache: bool,
        nologo: bool,
        private: bool,
        enhance: bool,
        safe: bool,
        transparent: bool,
        n: int,
        api_key: str,
        timeout: int = 120
    ) -> AsyncResult:
        if enhance is None:
            enhance = True if model == "flux" else False
        params = {
            "model": model,
            "nologo": str(nologo).lower(),
            "private": str(private).lower(),
            "enhance": str(enhance).lower(),
            "safe": str(safe).lower(),
        }
        if transparent:
            params["transparent"] = "true"
        image = [data for data, _ in media if isinstance(data, str) and data.startswith("http")] if media else []
        if image:
            params["image"] = ",".join(image)
        if alias in cls.video_models:
            params["aspectRatio"] = aspect_ratio
        elif model != "gptimage":
            params = use_aspect_ratio({
                "width": width,
                "height": height,
                **params
            }, "1:1" if aspect_ratio is None else aspect_ratio)
        query = "&".join(f"{k}={quote(str(v))}" for k, v in params.items() if v is not None)
        encoded_prompt = prompt.strip()
        if model == "gptimage" and aspect_ratio is not None:
            encoded_prompt = f"{encoded_prompt} aspect-ratio: {aspect_ratio}"
        encoded_prompt = quote_plus(encoded_prompt)[:4096 - len(cls.image_api_endpoint) - len(query) - 8].rstrip("%")
        if api_key and not api_key.startswith("g4f_") and not api_key.startswith("gfs_"):
            url = cls.gen_image_api_endpoint
        else:
            url = cls.image_api_endpoint
        url = url.format(f"{encoded_prompt}?{query}")

        def get_url_with_seed(i: int, seed: Optional[int] = None):
            if i == 0:
                if not cache and seed is None:
                    seed = random.randint(0, 2 ** 32)
            else:
                seed = random.randint(0, 2 ** 32)
            return f"{url}&seed={seed}" if seed else url

        headers = None
        if api_key:
            headers = {"authorization": f"Bearer {api_key}"}
        async with ClientSession(
            headers=DEFAULT_HEADERS,
            connector=get_connector(proxy=proxy),
            timeout=ClientTimeout(timeout)
        ) as session:
            responses = set()
            yield Reasoning(label=f"Generating {n} {('video' if alias in cls.video_models else 'image') + '' if n == 1 else 's'}")
            finished = 0
            start = time.time()

            async def get_image(responses: set, i: int, seed: Optional[int] = None):
                try:
                    async with session.get(get_url_with_seed(i, seed), allow_redirects=False,
                                           headers=headers) as response:
                        await raise_for_status(response)
                except Exception as e:
                    responses.add(e)
                    debug.error(f"Error fetching image:", e)
                if response.headers.get("x-error-type"):
                    responses.add(PreviewResponse(ImageResponse(str(response.url), prompt)))
                elif response.headers.get('content-type', '').startswith("image/"):
                    responses.add(ImageResponse(str(response.url), prompt, {"headers": headers}))
                elif response.headers.get('content-type', '').startswith("video/"):
                    responses.add(VideoResponse(str(response.url), prompt, {"headers": headers}))
                else:
                    responses.add(Exception(f"Unexpected content type: {response.headers.get('content-type')}"))

            tasks: list[asyncio.Task] = []
            for i in range(int(n)):
                tasks.append(asyncio.create_task(get_image(responses, i, seed)))
            while finished < n or len(responses) > 0:
                while len(responses) > 0:
                    item = responses.pop()
                    if isinstance(item, Exception):
                        if finished < 2:
                            yield Reasoning(status="")
                            for task in tasks:
                                task.cancel()
                            if cls.login_url in str(item):
                                raise MissingAuthError(item)
                            raise item
                        else:
                            finished += 1
                            yield Reasoning(
                                label=f"Image {finished}/{n} failed after {time.time() - start:.2f}s: {item}")
                    else:
                        finished += 1
                        yield Reasoning(label=f"Image {finished}/{n} generated in {time.time() - start:.2f}s")
                        yield item
                await asyncio.sleep(1)
            yield Reasoning(status="")
            await asyncio.gather(*tasks)

    @classmethod
    async def _generate_text(
        cls,
        model: str,
        messages: Messages,
        media: MediaListType,
        proxy: str,
        temperature: float,
        presence_penalty: float,
        top_p: float,
        frequency_penalty: float,
        response_format: Optional[dict],
        seed: Optional[int],
        cache: bool,
        stream: bool,
        extra_parameters: list[str],
        api_key: str,
        extra_body: dict,
        **kwargs
    ) -> AsyncResult:
        if not cache and seed is None:
            seed = random.randint(0, 2 ** 32)

        async with ClientSession(headers=DEFAULT_HEADERS, connector=get_connector(proxy=proxy)) as session:
            extra_body.update({param: kwargs[param] for param in extra_parameters if param in kwargs})
            if model in cls.audio_models:
                if "audio" in extra_body and extra_body.get("audio", {}).get("voice") is None:
                    extra_body["audio"]["voice"] = cls.default_voice
                elif "audio" not in extra_body:
                    extra_body["audio"] = {"voice": cls.default_voice}
                if extra_body.get("audio", {}).get("format") is None:
                    extra_body["audio"]["format"] = "mp3"
                    stream = False
                if "modalities" not in extra_body:
                    extra_body["modalities"] = ["text", "audio"]
            data = filter_none(
                messages=list(render_messages(messages, media)),
                model=model,
                temperature=temperature,
                presence_penalty=presence_penalty,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                response_format=response_format,
                stream=stream,
                seed=None if "tools" in extra_body else seed,
                **extra_body
            )
            if (not api_key or api_key.startswith("g4f_") or api_key.startswith("gfs_")) and cls.balance and cls.balance > 0:
                endpoint = cls.worker_api_endpoint
            elif api_key:
                endpoint = cls.gen_text_api_endpoint
            else:
                endpoint = cls.text_api_endpoint
            headers = None
            if api_key:
                headers = {"authorization": f"Bearer {api_key}"}
            yield JsonRequest.from_dict(data)
            async with session.post(endpoint, json=data, headers=headers) as response:
                if response.status in (400, 500):
                    debug.error(f"Error: {response.status} - Bad Request: {data}")
                async for chunk in read_response(response, stream, format_media_prompt(messages), cls.get_dict(),
                                                 kwargs.get("download_media", True)):
                    yield chunk