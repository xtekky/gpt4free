from __future__ import annotations

from typing import Optional

from .helper import format_media_prompt
from ..typing import AsyncResult, Messages, MediaListType
from ..config import STATIC_URL
from .PollinationsAI import PollinationsAI

class PollinationsImage(PollinationsAI):
    label = "PollinationsImage"
    parent = PollinationsAI.__name__
    default_model = "flux"
    default_vision_model = None
    default_image_model = default_model
    audio_models = {}

    @classmethod
    def get_models(cls, **kwargs):
        PollinationsAI.get_models()
        cls.image_models = PollinationsAI.image_models
        cls.models = cls.image_models
        return cls.models

    @classmethod
    def get_grouped_models(cls) -> dict[str, list[str]]:
        PollinationsAI.get_models()
        return [
            {"group": "Image Generation", "models": PollinationsAI.image_models},
        ]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        media: MediaListType = None,
        proxy: str = None,
        referrer: str = STATIC_URL,
        api_key: str = None,
        prompt: str = None,
        aspect_ratio: str = None,
        width: int = None,
        height: int = None,
        seed: Optional[int] = None,
        cache: bool = False,
        nologo: bool = True,
        private: bool = False,
        enhance: bool = False,
        safe: bool = False,
        transparent: bool = False,
        n: int = 1,
        **kwargs
    ) -> AsyncResult:
        # Calling model updates before creating a generator
        cls.get_models()
        async for chunk in cls._generate_image(
            model=model,
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
            transparent=transparent,
            n=n,
            referrer=referrer,
            api_key=api_key
        ):
            yield chunk
