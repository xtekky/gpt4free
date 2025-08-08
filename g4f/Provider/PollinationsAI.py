from __future__ import annotations

import time
import json
import random
import requests
import asyncio
from urllib.parse import quote, quote_plus
from typing import Optional
from aiohttp import ClientSession, ClientTimeout

from .helper import filter_none, format_media_prompt
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..typing import AsyncResult, Messages, MediaListType
from ..image import is_data_an_audio
from ..errors import ModelNotFoundError, MissingAuthError
from ..requests.raise_for_status import raise_for_status
from ..requests.aiohttp import get_connector
from ..image import use_aspect_ratio
from ..providers.response import ImageResponse, Reasoning, TitleGeneration, SuggestedFollowups
from ..tools.media import render_messages
from ..config import STATIC_URL
from .template.OpenaiTemplate import read_response
from .. import debug

DEFAULT_HEADERS = {
    "accept": "*/*",
    'accept-language': 'en-US,en;q=0.9',
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "referer": "https://pollinations.ai/",
    "origin": "https://pollinations.ai",
}

FOLLOWUPS_TOOLS = [{
    "type": "function",
    "function": {
        "name": "options",
        "description": "Provides options for the conversation",
        "parameters": {
            "properties": {
                "title": {
                    "title": "Conversation title. Prefixed with one or more emojies",
                    "type": "string"
                },
                "followups": {
                    "items": {
                        "type": "string"
                    },
                    "title": "Suggested 4 Followups (only user messages)",
                    "type": "array"
                }
            },
            "title": "Conversation",
            "type": "object"
        }
    }
}]

FOLLOWUPS_DEVELOPER_MESSAGE = [{
    "role": "developer",
    "content": "Provide conversation options.",
}]

class PollinationsAI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Pollinations AI"
    url = "https://pollinations.ai"
    login_url = "https://auth.pollinations.ai"
    active_by_default = True
    working = True
    supports_system_message = True
    supports_message_history = True

    # API endpoints
    text_api_endpoint = "https://text.pollinations.ai"
    openai_endpoint = "https://text.pollinations.ai/openai"
    image_api_endpoint = "https://image.pollinations.ai/"

    # Models configuration
    default_model = "openai"
    fallback_model = "deepseek"
    default_image_model = "flux"
    default_vision_model = default_model
    default_audio_model = "openai-audio"
    default_voice = "alloy"
    text_models = [default_model, "evil"]
    image_models = [default_image_model, "turbo", "kontext"]
    audio_models = {default_audio_model: []}
    vision_models = [default_vision_model]
    _models_loaded = False
    model_aliases = {
        "gpt-4.1-mini": "openai",
        "gpt-4.1-nano": "openai-fast",
        "gpt-4.1": "openai-large",
        "o4-mini": "openai-reasoning",
        "qwen-2.5-coder-32b": "qwen-coder",
        "llama-3.3-70b": "llama",
        "llama-4-scout": "llamascout",
        "mistral-small-3.1-24b": "mistral",
        "phi-4": "phi",
        "deepseek-r1": "deepseek-reasoning",
        "deepseek-v3-0324": "deepseek",
        "deepseek-v3": "deepseek",
        "grok-3-mini": "grok",
        "grok-3-mini-high": "grok",
        "gpt-4o-mini-audio": "openai-audio",
        "sdxl-turbo": "turbo",
        "gpt-image": "gptimage",
        "flux-dev": "flux",
        "flux-schnell": "flux",
        "flux-pro": "flux",
        "flux": "flux",
        "flux-kontext": "kontext",
    }
    swap_models = {value: key for key, value in model_aliases.items()}

    @classmethod
    def get_model(cls, model: str) -> str:
        """Get the internal model name from the user-provided model name."""
        if not model:
            return cls.default_model

        # Check if there's an alias for this model
        if model in cls.model_aliases:
            return cls.model_aliases[model]

        # Check if the model exists directly in our model lists
        if model in cls.text_models or model in cls.image_models or model in cls.audio_models:
            return model

        # If no match is found, raise an error
        raise ModelNotFoundError(f"PollinationsAI: Model {model} not found")

    @classmethod
    def get_models(cls, **kwargs):
        if not cls._models_loaded:
            try:
                # Update of image models
                image_response = requests.get("https://image.pollinations.ai/models")
                if image_response.ok:
                    new_image_models = image_response.json()
                else:
                    new_image_models = []

                # Combine image models without duplicates
                image_models = cls.image_models.copy()  # Start with default model
                
                # Add extra image models if not already in the list
                for model in new_image_models:
                    if model not in image_models:
                        image_models.append(model)
                
                cls.image_models = image_models

                text_response = requests.get("https://text.pollinations.ai/models")
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
                    cls.swap_models.get(model.get("name"), model.get("name"))
                    for model in models
                    if model.get("vision") and model not in cls.vision_models
                ])
                for alias, model in cls.model_aliases.items():
                    if model in cls.vision_models and alias not in cls.vision_models:
                        cls.vision_models.append(alias)

                # Create a set of unique text models starting with default model
                text_models = cls.text_models.copy()

                # Add models from the API response
                for model in models:
                    model_name = model.get("name")
                    if model_name and "input_modalities" in model and "text" in model["input_modalities"]:
                        text_models.append(cls.swap_models.get(model_name, model_name))

                # Convert to list and update text_models
                cls.text_models = list(dict.fromkeys(text_models))

                cls._models_loaded = True

            except Exception as e:
                # Save default models in case of an error
                if not cls.text_models:
                    cls.text_models = [cls.default_model]
                if not cls.image_models:
                    cls.image_models = [cls.default_image_model]
                debug.error(f"Failed to fetch models: {e}")

        # Return unique models across all categories
        all_models = cls.text_models.copy()
        all_models.extend(cls.image_models)
        all_models.extend(cls.audio_models.keys())
        if cls.default_audio_model in cls.audio_models:
            all_models.extend(cls.audio_models[cls.default_audio_model])
        return list(dict.fromkeys(all_models))

    @classmethod
    def get_grouped_models(cls) -> dict[str, list[str]]:
        cls.get_models()
        return [
            {"group": "Text Generation", "models": cls.text_models},
            {"group": "Image Generation", "models": cls.image_models},
            {"group": "Audio Generation", "models": list(cls.audio_models.keys())},
            {"group": "Audio Voices", "models": cls.audio_models.get(cls.default_audio_model, [])},
        ]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        proxy: str = None,
        cache: bool = None,
        referrer: str = STATIC_URL,
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
        extra_parameters: list[str] = ["tools", "parallel_tool_calls", "tool_choice", "reasoning_effort", "logit_bias", "voice", "modalities", "audio"],
        **kwargs
    ) -> AsyncResult:
        if cache is None:
            cache = kwargs.get("action") == "next"
        if extra_body is None:
            extra_body = {}
        if not model:
            has_audio = "audio" in kwargs or "audio" in kwargs.get("modalities", [])
            if not has_audio and media is not None:
                for media_data, filename in media:
                    if is_data_an_audio(media_data, filename):
                        has_audio = True
                        break
            model = cls.default_audio_model if has_audio else model
        try:
            model = cls.get_model(model) if model else None
        except ModelNotFoundError:
            pass
        if model in cls.image_models:
            async for chunk in cls._generate_image(
                model="gptimage" if model == "transparent" else model,
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
                referrer=referrer,
                api_key=api_key
            ):
                yield chunk
        else:
            if prompt is not None and len(messages) == 1:
                messages = [{
                    "role": "user",
                    "content": prompt
                }]
            if model and model in cls.audio_models[cls.default_audio_model]:
                kwargs["audio"] = {
                    "voice": model,
                }
                model = cls.default_audio_model
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
                referrer=referrer,
                api_key=api_key,
                extra_body=extra_body,
                **kwargs
            ):
                yield result

    @classmethod
    async def _generate_image(
        cls,
        model: str,
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
        referrer: str,
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
            "referrer": referrer
        }
        if transparent:
            params["transparent"] = "true"
        image = [data for data, _ in media if isinstance(data, str) and data.startswith("http")] if media else []
        if image:
            params["image"] = ",".join(image)
        if model != "gptimage":
            params = use_aspect_ratio({
                "width": width,
                "height": height,
                **params
            }, "1:1" if aspect_ratio is None else aspect_ratio)
        query = "&".join(f"{k}={quote(str(v))}" for k, v in params.items() if v is not None)
        encoded_prompt = prompt.strip(". \n")
        if model == "gptimage" and aspect_ratio is not None:
            encoded_prompt = f"{encoded_prompt} aspect-ratio: {aspect_ratio}"
        encoded_prompt = quote_plus(encoded_prompt)[:4096-len(cls.image_api_endpoint)-len(query)-8].rstrip("%")
        url = f"{cls.image_api_endpoint}prompt/{encoded_prompt}?{query}"
        def get_url_with_seed(i: int, seed: Optional[int] = None):
            if model == "gptimage":
                return url
            if i == 0:
                if not cache and seed is None:
                    seed = random.randint(0, 2**32)
            else:
                seed = random.randint(0, 2**32)
            return f"{url}&seed={seed}" if seed else url
        headers = {"referer": referrer}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        async with ClientSession(
            headers=DEFAULT_HEADERS,
            connector=get_connector(proxy=proxy),
            timeout=ClientTimeout(timeout)
        ) as session:
            responses = set()
            yield Reasoning(label=f"Generating {n} {'image' if n == 1 else 'images'}")
            finished = 0
            start = time.time()
            async def get_image(responses: set, i: int, seed: Optional[int] = None):
                try:
                    async with session.get(get_url_with_seed(i, seed), allow_redirects=False, headers=headers) as response:
                        await raise_for_status(response)
                except Exception as e:
                    responses.add(e)
                    debug.error(f"Error fetching image: {e}")
                responses.add(ImageResponse(str(response.url), prompt, {"headers": headers}))
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
        referrer: str,
        api_key: str,
        extra_body: dict,
        **kwargs
    ) -> AsyncResult:
        if not cache and seed is None:
            seed = random.randint(0, 2**32)

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
                seed=None if model =="grok" else seed,
                referrer=referrer,
                **extra_body
            )
            headers = {"referer": referrer}
            if api_key:
                headers["authorization"] = f"Bearer {api_key}"
            async with session.post(cls.openai_endpoint, json=data, headers=headers) as response:
                if response.status in (400, 500):
                    debug.error(f"Error: {response.status} - Bad Request: {data}")
                full_resposne = []
                async for chunk in read_response(response, stream, format_media_prompt(messages), cls.get_dict(), kwargs.get("download_media", True)):
                    if isinstance(chunk, str):
                        full_resposne.append(chunk)
                    yield chunk
                if full_resposne:
                    full_content = "".join(full_resposne)
                    if kwargs.get("action") == "next" and model != "evil":
                        tool_messages = []
                        for message in messages:
                            if message.get("role") == "user":
                                if isinstance(message.get("content"), str):
                                    tool_messages.append({"role": "user", "content": message.get("content")})
                                elif isinstance(message.get("content"), list):
                                    next_value = message.get("content").pop()
                                    if isinstance(next_value, dict):
                                        next_value = next_value.get("text")
                                        if next_value:
                                            tool_messages.append({"role": "user", "content": next_value})
                        tool_messages.append({"role": "assistant", "content": full_content})
                        data = {
                            "model": "openai",
                            "messages": tool_messages + FOLLOWUPS_DEVELOPER_MESSAGE,
                            "tool_choice": "required",
                            "tools": FOLLOWUPS_TOOLS
                        }
                        async with session.post(cls.openai_endpoint, json=data, headers=headers) as response:
                            try:
                                await raise_for_status(response)
                                tool_calls = (await response.json()).get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
                                if tool_calls:
                                    arguments = json.loads(tool_calls.pop().get("function", {}).get("arguments"))
                                    if arguments.get("title"):
                                        yield TitleGeneration(arguments.get("title"))
                                    if arguments.get("followups"):
                                        yield SuggestedFollowups(arguments.get("followups"))
                            except Exception as e:
                                debug.error("Error generating title and followups:", e)