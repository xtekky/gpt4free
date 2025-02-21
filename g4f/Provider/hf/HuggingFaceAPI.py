from __future__ import annotations

from ...providers.types import Messages
from ...typing import ImagesType
from ...requests import StreamSession, raise_for_status
from ...errors import ModelNotSupportedError
from ...providers.helper import get_last_user_message
from ..template.OpenaiTemplate import OpenaiTemplate
from .models import model_aliases, vision_models, default_vision_model
from .HuggingChat import HuggingChat
from ... import debug

class HuggingFaceAPI(OpenaiTemplate):
    label = "HuggingFace (Inference API)"
    parent = "HuggingFace"
    url = "https://api-inference.huggingface.com"
    api_base = "https://api-inference.huggingface.co/v1"
    working = True
    needs_auth = True

    default_model = default_vision_model
    default_vision_model = default_vision_model
    vision_models = vision_models
    model_aliases = model_aliases

    pipeline_tags: dict[str, str] = {}

    @classmethod
    def get_models(cls, **kwargs):
        if not cls.models:
            HuggingChat.get_models()
            cls.models = HuggingChat.text_models.copy()
            for model in cls.vision_models:
                if model not in cls.models:
                    cls.models.append(model)
        return cls.models

    @classmethod
    async def get_pipline_tag(cls, model: str, api_key: str = None):
        if model in cls.pipeline_tags:
            return cls.pipeline_tags[model]
        async with StreamSession(
            timeout=30,
            headers=cls.get_headers(False, api_key),
        ) as session:
            async with session.get(f"https://huggingface.co/api/models/{model}") as response:
                await raise_for_status(response)
                model_data = await response.json()
                cls.pipeline_tags[model] = model_data.get("pipeline_tag")
        return cls.pipeline_tags[model]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_base: str = None,
        api_key: str = None,
        max_tokens: int = 2048,
        max_inputs_lenght: int = 10000,
        images: ImagesType = None,
        **kwargs
    ):
        if model in cls.model_aliases:
            model = cls.model_aliases[model]
        api_base = f"https://api-inference.huggingface.co/models/{model}/v1"
        pipeline_tag = await cls.get_pipline_tag(model, api_key)
        if pipeline_tag not in ("text-generation", "image-text-to-text"):
            raise ModelNotSupportedError(f"Model is not supported: {model} in: {cls.__name__} pipeline_tag: {pipeline_tag}")
        elif images and  pipeline_tag != "image-text-to-text":
            raise ModelNotSupportedError(f"Model does not support images: {model} in: {cls.__name__} pipeline_tag: {pipeline_tag}")
        start = calculate_lenght(messages)
        if start > max_inputs_lenght:
            if len(messages) > 6:
                messages = messages[:3] + messages[-3:]
            if calculate_lenght(messages) > max_inputs_lenght:
                last_user_message = [{"role": "user", "content": get_last_user_message(messages)}]
                if len(messages) > 2:
                    messages = [m for m in messages if m["role"] == "system"] + last_user_message
                if len(messages) > 1 and calculate_lenght(messages) > max_inputs_lenght:
                    messages = last_user_message
            debug.log(f"Messages trimmed from: {start} to: {calculate_lenght(messages)}")
        async for chunk in super().create_async_generator(model, messages, api_base=api_base, api_key=api_key, max_tokens=max_tokens, images=images, **kwargs):
            yield chunk

def calculate_lenght(messages: Messages) -> int:
    return sum([len(message["content"]) + 16 for message in messages])