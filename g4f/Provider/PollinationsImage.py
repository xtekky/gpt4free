from __future__ import annotations

from typing import Optional

from .helper import format_image_prompt
from ..typing import AsyncResult, Messages
from .PollinationsAI import PollinationsAI

class PollinationsImage(PollinationsAI):
    label = "PollinationsImage"
    default_model = "flux"
    default_vision_model = None
    default_image_model = default_model
    image_models = [default_image_model]  # Default models
    _models_loaded = False  # Add a checkbox for synchronization

    @classmethod
    def get_models(cls, **kwargs):
        if not cls._models_loaded:
            # Calling the parent method to load models
            super().get_models()
            # Combine models from the parent class and additional ones
            all_image_models = list(dict.fromkeys(
                cls.image_models +
                PollinationsAI.image_models +
                cls.extra_image_models
            ))
            cls.image_models = all_image_models
            cls._models_loaded = True
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
        cache: bool = False,
        nologo: bool = True,
        private: bool = False,
        enhance: bool = False,
        safe: bool = False,
        **kwargs
    ) -> AsyncResult:
        # Calling model updates before creating a generator
        cls.get_models()
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
