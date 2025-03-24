from __future__ import annotations

import json
import random
import requests
from urllib.parse import quote_plus
from typing import Optional
from aiohttp import ClientSession

from .helper import filter_none, format_image_prompt
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..typing import AsyncResult, Messages, MediaListType
from ..image import to_data_uri, is_data_an_audio, to_input_audio
from ..errors import ModelNotFoundError
from ..requests.raise_for_status import raise_for_status
from ..requests.aiohttp import get_connector
from ..image.copy_images import save_response_media
from ..image import use_aspect_ratio
from ..providers.response import FinishReason, Usage, ToolCalls, ImageResponse
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
    text_models = [default_model]
    image_models = [default_image_model]
    extra_image_models = ["flux-pro", "flux-dev", "flux-schnell", "midjourney", "dall-e-3"]
    vision_models = [default_vision_model, "gpt-4o-mini", "o3-mini", "openai", "openai-large"]
    extra_text_models = vision_models
    _models_loaded = False
    model_aliases = {
        ### Text Models ###
        "gpt-4o-mini": "openai",
        "gpt-4": "openai-large",
        "gpt-4o": "openai-large",
        "o3-mini": "openai-reasoning",
        "qwen-2.5-coder-32b": "qwen-coder",
        "llama-3.3-70b": "llama",
        "mistral-nemo": "mistral",
        "gpt-4o-mini": "searchgpt",
        "llama-3.1-8b": "llamalight",
        "llama-3.3-70b": "llama-scaleway",
        "phi-4": "phi",
        "gemini-2.0": "gemini",
        "gemini-2.0-flash": "gemini",
        "gemini-2.0-flash-thinking": "gemini-thinking",
        "deepseek-r1": "deepseek-r1-llama",
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

                # Combine models without duplicates
                all_image_models = (
                    cls.image_models +  # Already contains the default
                    cls.extra_image_models + 
                    new_image_models
                )
                cls.image_models = list(dict.fromkeys(all_image_models))

                # Update of text models
                text_response = requests.get("https://text.pollinations.ai/models")
                text_response.raise_for_status()
                models = text_response.json()
                original_text_models = [
                    model.get("name") 
                    for model in models
                    if model.get("type") == "chat"
                ]
                cls.audio_models = {
                    model.get("name"): model.get("voices")
                    for model in models
                    if model.get("audio")
                }
                
                # Combining text models
                combined_text = (
                    cls.text_models +  # Already contains the default
                    cls.extra_text_models + 
                    [
                        model for model in original_text_models
                        if model not in cls.extra_text_models
                    ]
                )
                cls.text_models = list(dict.fromkeys(combined_text))
                
                cls._models_loaded = True

            except Exception as e:
                # Save default models in case of an error
                if not cls.text_models:
                    cls.text_models = [cls.default_model]
                if not cls.image_models:
                    cls.image_models = [cls.default_image_model]
                debug.error(f"Failed to fetch models: {e}")

        return cls.text_models + cls.image_models

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
        # Text generation parameters
        media: MediaListType = None,
        temperature: float = None,
        presence_penalty: float = None,
        top_p: float = 1,
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
                safe=safe
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
        safe: bool
    ) -> AsyncResult:
        if not cache and seed is None:
            seed = random.randint(9999, 99999999)
        params = use_aspect_ratio({
            "seed": seed,
            "width": width,
            "height": height,
            "model": model,
            "nologo": str(nologo).lower(),
            "private": str(private).lower(),
            "enhance": str(enhance).lower(),
            "safe": str(safe).lower()
        }, aspect_ratio)
        query = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items() if v is not None)
        url = f"{cls.image_api_endpoint}prompt/{quote_plus(prompt)}?{query}"

        async with ClientSession(headers=DEFAULT_HEADERS, connector=get_connector(proxy=proxy)) as session:
            async with session.get(url, allow_redirects=True) as response:
                await raise_for_status(response)
                yield ImageResponse(url, prompt)

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
            seed = random.randint(9999, 99999999)
        json_mode = False
        if response_format and response_format.get("type") == "json_object":
            json_mode = True

        if media and messages:
            last_message = messages[-1].copy()
            image_content = [
                {
                    "type": "input_audio",
                    "input_audio": to_input_audio(media_data, filename)
                }
                if is_data_an_audio(media_data, filename) else {
                    "type": "image_url",
                    "image_url": {"url": to_data_uri(media_data)}
                }
                for media_data, filename in media
            ]
            last_message["content"] = image_content + ([{"type": "text", "text": last_message["content"]}] if isinstance(last_message["content"], str) else image_content)
            messages[-1] = last_message

        async with ClientSession(headers=DEFAULT_HEADERS, connector=get_connector(proxy=proxy)) as session:
            if model in cls.audio_models:
                #data["voice"] = random.choice(cls.audio_models[model])
                url = cls.text_api_endpoint
                stream = False
            else:
                url = cls.openai_endpoint
            extra_parameters = {param: kwargs[param] for param in extra_parameters if param in kwargs}
            data = filter_none(**{
                "messages": messages,
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
                async for chunk in save_response_media(response, messages[-1]["content"]):
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
                            choices = result.get("choices", [{}])
                            choice = choices.pop() if choices else {}
                            content = choice.get("delta", {}).get("content")
                            if content:
                                yield content
                            if "usage" in result:
                                yield Usage(**result["usage"])
                            finish_reason = choice.get("finish_reason")
                            if finish_reason:
                                yield FinishReason(finish_reason)
                    return
                result = await response.json()
                choice = result["choices"][0]
                message = choice.get("message", {})
                content = message.get("content", "")

                if "tool_calls" in message:
                    yield ToolCalls(message["tool_calls"])

                if content:
                    yield content

                if "usage" in result:
                    yield Usage(**result["usage"])

                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    yield FinishReason(finish_reason)
