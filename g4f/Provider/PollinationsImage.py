from __future__ import annotations

from typing import Optional

from .helper import format_image_prompt
from ..typing import AsyncResult, Messages
from .PollinationsAI import PollinationsAI

class PollinationsImage(PollinationsAI):
    default_model = "flux"
    default_vision_model = None
    default_image_model = default_model

    @classmethod
    def get_models(cls, **kwargs):
        if not cls.image_models:
            cls.image_models = list(dict.fromkeys([*cls.image_models, *cls.extra_image_models]))
        return cls.image_models

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
        **kwargs
    ) -> AsyncResult:
        async for chunk in cls._generate_image(
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
        ):
            yield chunk