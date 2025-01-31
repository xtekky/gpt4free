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
from ..requests.raise_for_status import raise_for_status
from ..requests.aiohttp import get_connector
from ..providers.response import ImageResponse, FinishReason, Usage, Reasoning

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
    extra_image_models = ["midjourney", "dall-e-3", "flux-pro"]
    vision_models = [default_vision_model, "gpt-4o-mini"]
    reasoning_models = ['deepseek-reasoner', 'deepseek-r1']
    extra_text_models = ["claude", "claude-email", "p1"] + vision_models + reasoning_models
    model_aliases = {
        ### Text Models ###
        "gpt-4o-mini": "openai",
        "gpt-4": "openai-large",
        "gpt-4o": "openai-large",
        "qwen-2.5-72b": "qwen",
        "qwen-2.5-coder-32b": "qwen-coder",
        "llama-3.3-70b": "llama",
        "mistral-nemo": "mistral",
        "gpt-4o-mini": "rtist",
        "gpt-4o": "searchgpt",
        "gpt-4o-mini": "p1",
        "deepseek-chat": "deepseek",
        "deepseek-chat": "claude-hybridspace",
        "llama-3.1-8b": "llamalight",
        "gpt-4o-vision": "gpt-4o",
        "gpt-4o-mini-vision": "gpt-4o-mini",
        "gpt-4o-mini": "claude",
        "deepseek-chat": "claude-email",
        "deepseek-r1": "deepseek-reasoner",
        
        ### Image Models ###
        "sdxl-turbo": "turbo", 
    }
    text_models = []

    @classmethod
    def get_models(cls, **kwargs):
        if not cls.image_models:
            url = "https://image.pollinations.ai/models"
            response = requests.get(url)
            raise_for_status(response)
            cls.image_models = response.json()
            cls.image_models = list(dict.fromkeys([*cls.image_models, *cls.extra_image_models]))
        
        if not cls.text_models:
            url = "https://text.pollinations.ai/models"
            response = requests.get(url)
            raise_for_status(response)
            original_text_models = [model.get("name") for model in response.json()]
            combined_text = cls.extra_text_models + [
                model for model in original_text_models 
                if model not in cls.extra_text_models
            ]
            cls.text_models = list(dict.fromkeys(combined_text))
        
        return list(dict.fromkeys([*cls.text_models, *cls.image_models]))

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
        model = cls.get_model(model)
        if not cache and seed is None:
            seed = random.randint(0, 100000)

        if model in cls.image_models:
           yield await cls._generate_image(
                model=model,
                prompt=format_image_prompt(messages, prompt),
                proxy=proxy,
                width=width,
                height=height,
                seed=seed,
                nologo=nologo,
                private=private,
                enhance=enhance,
                safe=safe
            )
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
        nologo: bool,
        private: bool,
        enhance: bool,
        safe: bool
    ) -> ImageResponse:
        params = {
            "seed": seed,
            "width": width,
            "height": height,
            "model": model,
            "nologo": nologo,
            "private": private,
            "enhance": enhance,
            "safe": safe
        }
        params = {k: json.dumps(v) if isinstance(v, bool) else v for k, v in params.items() if v is not None}
        async with ClientSession(headers=DEFAULT_HEADERS, connector=get_connector(proxy=proxy)) as session:
            async with session.head(f"{cls.image_api_endpoint}prompt/{quote_plus(prompt)}", params=params) as response:
                await raise_for_status(response)
                return ImageResponse(str(response.url), prompt)

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
        jsonMode = False
        if response_format is not None and "type" in response_format:
            if response_format["type"] == "json_object":
                jsonMode = True

        if images is not None and messages:
            last_message = messages[-1].copy()
            last_message["content"] = [
                *[{
                    "type": "image_url",
                    "image_url": {"url": to_data_uri(image)}
                } for image, _ in images],
                {
                    "type": "text",
                    "text": messages[-1]["content"]
                }
            ]
            messages[-1] = last_message

        async with ClientSession(headers=DEFAULT_HEADERS, connector=get_connector(proxy=proxy)) as session:
            data = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "presence_penalty": presence_penalty,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "jsonMode": jsonMode,
                "stream": False,
                "seed": seed,
                "cache": cache
            }
            async with session.post(cls.text_api_endpoint, json=filter_none(**data)) as response:
                await raise_for_status(response)
                async for line in response.content:
                    decoded_chunk = line.decode(errors="replace")
                    if "data: [DONE]" in decoded_chunk:
                        break
                    try:
                        json_str = decoded_chunk.replace("data:", "").strip()
                        data = json.loads(json_str)
                        choice = data["choices"][0]
                        message = choice.get("message") or choice.get("delta", {})
                        
                        # Handle reasoning content
                        if model in cls.reasoning_models:
                            if "reasoning_content" in message:
                                yield Reasoning(status=message["reasoning_content"].strip())
                        
                        if "usage" in data:
                            yield Usage(**data["usage"])
                        content = message.get("content", "")
                        if content:
                            yield content.replace("\\(", "(").replace("\\)", ")")
                        if "finish_reason" in choice and choice["finish_reason"]:
                            yield FinishReason(choice["finish_reason"])
                            break
                    except json.JSONDecodeError:
                        yield decoded_chunk.strip()
                    except Exception as e:
                        yield FinishReason("error")
                        break
