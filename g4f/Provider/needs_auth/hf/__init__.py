from __future__ import annotations

import random

from ....typing import AsyncResult, Messages
from ....providers.response import ImageResponse
from ....errors import ModelNotFoundError, MissingAuthError
from ...base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .HuggingChat import HuggingChat
from .HuggingFaceAPI import HuggingFaceAPI
from .HuggingFaceInference import HuggingFaceInference
from .HuggingFaceMedia import HuggingFaceMedia
from .models import model_aliases, image_model_aliases, vision_models, default_model
from .... import debug

class HuggingFace(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://huggingface.co"
    login_url = "https://huggingface.co/settings/tokens"
    working = True
    active_by_default = True

    @classmethod
    def get_models(cls, **kwargs) -> list[str]:
        if not cls.models:
            cls.models = HuggingFaceInference.get_models()
            cls.image_models = HuggingFaceInference.image_models
        return cls.models

    model_aliases = {**model_aliases, **image_model_aliases}
    vision_models = vision_models
    default_model = default_model

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        **kwargs
    ) -> AsyncResult:
        if model in cls.model_aliases:
            model = cls.model_aliases[model]
        # if "tools" not in kwargs and "media" not in kwargs and random.random() >= 0.5:
        #     try:
        #         is_started = False
        #         async for chunk in HuggingFaceInference.create_async_generator(model, messages, **kwargs):
        #             if isinstance(chunk, (str, ImageResponse)):
        #                 is_started = True
        #             yield chunk
        #         if is_started:
        #             return
        #     except Exception as e:
        #         if is_started:
        #             raise e
        #         debug.error(f"{cls.__name__} {type(e).__name__}; {e}")
        if not cls.image_models:
            cls.get_models()
        try:
            async for chunk in HuggingFaceMedia.create_async_generator(model, messages, **kwargs):
                yield chunk
            return
        except ModelNotFoundError:
            pass
        # if model in cls.image_models:
        #     if "api_key" not in kwargs:
        #         async for chunk in HuggingChat.create_async_generator(model, messages, **kwargs):
        #             yield chunk
        #     else:
        #         async for chunk in HuggingFaceInference.create_async_generator(model, messages, **kwargs):
        #             yield chunk
        #     return
        try:
            async for chunk in HuggingFaceAPI.create_async_generator(model, messages, **kwargs):
                yield chunk
        except (ModelNotFoundError, MissingAuthError):
            async for chunk in HuggingFaceInference.create_async_generator(model, messages, **kwargs):
                yield chunk
