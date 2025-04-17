from __future__ import annotations

import json
import random
import requests
import asyncio
from urllib.parse import quote_plus
from typing import Optional
from aiohttp import ClientSession

from .helper import filter_none, format_image_prompt
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..typing import AsyncResult, Messages, MediaListType
from ..image import is_data_an_audio
from ..errors import ModelNotFoundError, ResponseError
from ..requests.raise_for_status import raise_for_status
from ..requests.aiohttp import get_connector
from ..image.copy_images import save_response_media
from ..image import use_aspect_ratio
from ..providers.response import FinishReason, Usage, ToolCalls, ImageResponse
from ..tools.media import render_messages
from .. import debug

DEFAULT_HEADERS = {
    "accept": "*/*",
    'accept-language': 'en-US,en;q=0.9',
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "referer": "https://pollinations.ai/",
    "origin": "https://pollinations.ai",
}

class PollinationsAI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Pollinations AI"
    url = "https://pollinations.ai"

    working = True
    supports_system_message = True
    supports_message_history = True

    # API endpoints
    text_api_endpoint = "https://text.pollinations.ai"
    openai_endpoint = "https://text.pollinations.ai/openai"
    image_api_endpoint = "https://image.pollinations.ai/"

    # Models configuration
    default_model = "openai"
    default_image_model = "flux"
    default_vision_model = default_model
    default_audio_model = "openai-audio"
    text_models = [default_model]
    image_models = [default_image_model]
    audio_models = [default_audio_model]
    extra_image_models = ["flux-pro", "flux-dev", "flux-schnell", "midjourney", "dall-e-3", "turbo"]
    vision_models = [default_vision_model, "gpt-4o-mini", "openai", "openai-large", "searchgpt"]
    _models_loaded = False
    # https://github.com/pollinations/pollinations/blob/master/text.pollinations.ai/generateTextPortkey.js#L15
    model_aliases = {
        ### Text Models ###
        "gpt-4o-mini": "openai",
        "gpt-4": "openai-large",
        "gpt-4o": "openai-large",
        "qwen-2.5-coder-32b": "qwen-coder",
        "llama-3.3-70b": "llama",
        "llama-4-scout": "llamascout",
        "mistral-nemo": "mistral",
        "llama-3.1-8b": "llamalight",
        "llama-3.3-70b": "llama-scaleway",
        "phi-4": "phi",
        "gemini-2.0": "gemini",
        "gemini-2.0-flash": "gemini",
        "gemini-2.0-flash-thinking": "gemini-thinking",
        "deepseek-r1": "deepseek-reasoning-large",
        "deepseek-r1": "deepseek-reasoning",
        "deepseek-v3": "deepseek",
        "llama-3.2-11b": "llama-vision",
        "gpt-4o-audio": "openai-audio",
        
        ### Image Models ###
        "sdxl-turbo": "turbo",
    }

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
                all_image_models = [cls.default_image_model]  # Start with default model
                
                # Add extra image models if not already in the list
                for model in cls.extra_image_models + new_image_models:
                    if model not in all_image_models:
                        all_image_models.append(model)
                
                cls.image_models = all_image_models

                text_response = requests.get("https://text.pollinations.ai/models")
                text_response.raise_for_status()
                models = text_response.json()

                # Purpose of audio models
                cls.audio_models = {
                    model.get("name"): model.get("voices")
                    for model in models
                    if "output_modalities" in model and "audio" in model["output_modalities"]
                }

                # Create a set of unique text models starting with default model
                unique_text_models = cls.text_models.copy()

                # Add models from vision_models
                unique_text_models.extend(cls.vision_models)

                # Add models from the API response
                for model in models:
                    model_name = model.get("name")
                    if model_name and "input_modalities" in model and "text" in model["input_modalities"]:
                        unique_text_models.append(model_name)

                # Convert to list and update text_models
                cls.text_models = list(dict.fromkeys(unique_text_models))

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
        return list(dict.fromkeys(all_models))

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        proxy: str = None,
        cache: bool = False,
        # Image generation parameters
        prompt: str = None,
        aspect_ratio: str = "1:1",
        width: int = None,
        height: int = None,
        seed: Optional[int] = None,
        nologo: bool = True,
        private: bool = False,
        enhance: bool = False,
        safe: bool = False,
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
        # Load model list
        cls.get_models()
        if not model:
            has_audio = "audio" in kwargs
            if not has_audio and media is not None:
                for media_data, filename in media:
                    if is_data_an_audio(media_data, filename):
                        has_audio = True
                        break
            model = next(iter(cls.audio_models)) if has_audio else model
        try:
            model = cls.get_model(model)
        except ModelNotFoundError:
            if model not in cls.image_models:
                raise

        if model in cls.image_models:
            async for chunk in cls._generate_image(
                model=model,
                prompt=format_image_prompt(messages, prompt),
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
                n=n
            ):
                yield chunk
        else:
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
                **kwargs
            ):
                yield result

    @classmethod
    async def _generate_image(
        cls,
        model: str,
        prompt: str,
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
        n: int
    ) -> AsyncResult:
        params = use_aspect_ratio({
            "width": width,
            "height": height,
            "model": model,
            "nologo": str(nologo).lower(),
            "private": str(private).lower(),
            "enhance": str(enhance).lower(),
            "safe": str(safe).lower()
        }, aspect_ratio)
        query = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items() if v is not None)
        prompt = quote_plus(prompt)[:2048-256-len(query)]
        url = f"{cls.image_api_endpoint}prompt/{prompt}?{query}"
        def get_image_url(i: int, seed: Optional[int] = None):
            if i == 1:
                if not cache and seed is None:
                    seed = random.randint(0, 2**32)
            else:
                seed = random.randint(0, 2**32)
            return f"{url}&seed={seed}" if seed else url
        async with ClientSession(headers=DEFAULT_HEADERS, connector=get_connector(proxy=proxy)) as session:
            async def get_image(i: int, seed: Optional[int] = None):
                async with session.get(get_image_url(i, seed), allow_redirects=False) as response:
                    try:
                        await raise_for_status(response)
                    except Exception as e:
                        debug.error(f"Error fetching image: {e}")
                        return str(response.url)
                    return str(response.url)
            yield ImageResponse(await asyncio.gather(*[
                get_image(i, seed) for i in range(int(n))
            ]), prompt)

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
        **kwargs
    ) -> AsyncResult:
        if not cache and seed is None:
            seed = random.randint(0, 2**32)
        json_mode = False
        if response_format and response_format.get("type") == "json_object":
            json_mode = True

        async with ClientSession(headers=DEFAULT_HEADERS, connector=get_connector(proxy=proxy)) as session:
            if model in cls.audio_models:
                url = cls.text_api_endpoint
                stream = False
            else:
                url = cls.openai_endpoint
            extra_parameters = {param: kwargs[param] for param in extra_parameters if param in kwargs}
            data = filter_none(**{
                "messages": list(render_messages(messages, media)),
                "model": model,
                "temperature": temperature,
                "presence_penalty": presence_penalty,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "jsonMode": json_mode,
                "stream": stream,
                "seed": seed,
                "cache": cache,
                **extra_parameters
            })
            async with session.post(url, json=data) as response:
                await raise_for_status(response)
                async for chunk in save_response_media(response, format_image_prompt(messages), [model]):
                    yield chunk
                    return
                if response.headers["content-type"].startswith("text/plain"):
                    yield await response.text()
                    return
                elif response.headers["content-type"].startswith("text/event-stream"):
                    async for line in response.content:
                        if line.startswith(b"data: "):
                            if line[6:].startswith(b"[DONE]"):
                                break
                            result = json.loads(line[6:])
                            if "error" in result:
                                raise ResponseError(result["error"].get("message", result["error"]))
                            if "usage" in result:
                                yield Usage(**result["usage"])
                            choices = result.get("choices", [{}])
                            choice = choices.pop() if choices else {}
                            content = choice.get("delta", {}).get("content")
                            if content:
                                yield content
                            finish_reason = choice.get("finish_reason")
                            if finish_reason:
                                yield FinishReason(finish_reason)
                    return
                result = await response.json()
                if "choices" in result:
                    choice = result["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    if content:
                        yield content
                    if "tool_calls" in message:
                        yield ToolCalls(message["tool_calls"])
                else:
                    raise ResponseError(result)
                if "usage" in result:
                    yield Usage(**result["usage"])
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    yield FinishReason(finish_reason)
