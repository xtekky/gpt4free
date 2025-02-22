from __future__ import annotations

import json
import random
import requests
from urllib.parse import quote_plus
from typing import Optional
from aiohttp import ClientSession

from .helper import filter_none, format_image_prompt
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..typing import AsyncResult, Messages, ImagesType
from ..image import to_data_uri
from ..errors import ModelNotFoundError
from ..requests.raise_for_status import raise_for_status
from ..requests.aiohttp import get_connector
from ..providers.response import ImageResponse, ImagePreview, FinishReason, Usage, Reasoning

DEFAULT_HEADERS = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
}

class PollinationsAI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Pollinations AI"
    url = "https://pollinations.ai"

    working = True
    supports_stream = False
    supports_system_message = True
    supports_message_history = True

    # API endpoints
    text_api_endpoint = "https://text.pollinations.ai/openai"
    image_api_endpoint = "https://image.pollinations.ai/"

    # Models configuration
    default_model = "openai"
    default_image_model = "flux"
    default_vision_model = "gpt-4o"
    image_models = ["flux-pro", "flux-dev", "flux-schnell", "midjourney", "dall-e-3", "turbo"]
    vision_models = [default_vision_model, "gpt-4o-mini"]
    extra_text_models = ["claude", "claude-email", "deepseek-reasoner", "deepseek-r1"] + vision_models
    model_aliases = {
        ### Text Models ###
        "gpt-4o-mini": "openai",
        "gpt-4": "openai-large",
        "gpt-4o": "openai-large",
        "qwen-2.5-coder-32b": "qwen-coder",
        "llama-3.3-70b": "llama",
        "mistral-nemo": "mistral",
        "gpt-4o-mini": "rtist",
        "gpt-4o": "searchgpt",
        "gpt-4o-mini": "p1",
        "deepseek-chat": "claude-hybridspace",
        "llama-3.1-8b": "llamalight",
        "gpt-4o-vision": "gpt-4o",
        "gpt-4o-mini-vision": "gpt-4o-mini",
        "gpt-4o-mini": "claude",
        "deepseek-chat": "claude-email",
        "deepseek-r1": "deepseek-reasoner",
        "gemini-2.0-flash": "gemini",
        "gemini-2.0-flash-thinking": "gemini-thinking",
        
        ### Image Models ###
        "sdxl-turbo": "turbo",
    }
    text_models = []

    @classmethod
    def get_models(cls, **kwargs):
        if not cls.text_models or not cls.image_models:
            try:
                image_response = requests.get("https://image.pollinations.ai/models")
                image_response.raise_for_status()
                new_image_models = image_response.json()
                cls.image_models = list(dict.fromkeys([*cls.image_models, *new_image_models]))
                
                text_response = requests.get("https://text.pollinations.ai/models")
                text_response.raise_for_status()
                original_text_models = [model.get("name") for model in text_response.json()]
                
                combined_text = cls.extra_text_models + [
                    model for model in original_text_models 
                    if model not in cls.extra_text_models
                ]
                cls.text_models = list(dict.fromkeys(combined_text))
            except Exception as e:
                raise RuntimeError(f"Failed to fetch models: {e}") from e
            
        return cls.text_models + cls.image_models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        prompt: str = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        nologo: bool = True,
        private: bool = False,
        enhance: bool = False,
        safe: bool = False,
        images: ImagesType = None,
        temperature: float = None,
        presence_penalty: float = None,
        top_p: float = 1,
        frequency_penalty: float = None,
        response_format: Optional[dict] = None,
        cache: bool = False,
        **kwargs
    ) -> AsyncResult:
        if images is not None and not model:
            model = cls.default_vision_model
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
                images=images,
                proxy=proxy,
                temperature=temperature,
                presence_penalty=presence_penalty,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                response_format=response_format,
                seed=seed,
                cache=cache,
            ):
                yield result

    @classmethod
    async def _generate_image(
        cls,
        model: str,
        prompt: str,
        proxy: str,
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
        params = {
            "seed": str(seed) if seed is not None else None,
            "width": str(width),
            "height": str(height),
            "model": model,
            "nologo": str(nologo).lower(),
            "private": str(private).lower(),
            "enhance": str(enhance).lower(),
            "safe": str(safe).lower()
        }
        params = {k: v for k, v in params.items() if v is not None}
        query = "&".join(f"{k}={quote_plus(v)}" for k, v in params.items())
        prefix = f"{model}_{seed}" if seed is not None else model
        url = f"{cls.image_api_endpoint}prompt/{prefix}_{quote_plus(prompt)}?{query}"
        yield ImagePreview(url, prompt)

        async with ClientSession(headers=DEFAULT_HEADERS, connector=get_connector(proxy=proxy)) as session:
            async with session.get(url, allow_redirects=True) as response:
                await raise_for_status(response)
                image_url = str(response.url)
                yield ImageResponse(image_url, prompt)

    @classmethod
    async def _generate_text(
        cls,
        model: str,
        messages: Messages,
        images: Optional[ImagesType],
        proxy: str,
        temperature: float,
        presence_penalty: float,
        top_p: float,
        frequency_penalty: float,
        response_format: Optional[dict],
        seed: Optional[int],
        cache: bool
    ) -> AsyncResult:
        if not cache and seed is None:
            seed = random.randint(9999, 99999999)
        json_mode = False
        if response_format and response_format.get("type") == "json_object":
            json_mode = True

        if images and messages:
            last_message = messages[-1].copy()
            image_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": to_data_uri(image)}
                }
                for image, _ in images
            ]
            last_message["content"] = image_content + [{"type": "text", "text": last_message["content"]}]
            messages[-1] = last_message

        async with ClientSession(headers=DEFAULT_HEADERS, connector=get_connector(proxy=proxy)) as session:
            data = filter_none(**{
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "presence_penalty": presence_penalty,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "jsonMode": json_mode,
                "stream": False,
                "seed": seed,
                "cache": cache
            })
            
            async with session.post(cls.text_api_endpoint, json=data) as response:
                await raise_for_status(response)
                result = await response.json()
                choice = result["choices"][0]
                message = choice.get("message", {})
                content = message.get("content", "")
                
                if content:
                    yield content.replace("\\(", "(").replace("\\)", ")")
                
                if "usage" in result:
                    yield Usage(**result["usage"])
                
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    yield FinishReason(finish_reason)
