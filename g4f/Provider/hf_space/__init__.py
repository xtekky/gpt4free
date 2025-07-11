from __future__ import annotations

import random

from ...typing import AsyncResult, Messages, MediaListType
from ...errors import ResponseError
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin

from .BlackForestLabs_Flux1Dev       import BlackForestLabs_Flux1Dev
from .BlackForestLabs_Flux1KontextDev import BlackForestLabs_Flux1KontextDev
from .CohereForAI_C4AI_Command       import CohereForAI_C4AI_Command
from .DeepseekAI_JanusPro7b          import DeepseekAI_JanusPro7b
from .Microsoft_Phi_4_Multimodal     import Microsoft_Phi_4_Multimodal
from .Qwen_Qwen_2_5                  import Qwen_Qwen_2_5
from .Qwen_Qwen_2_5M                 import Qwen_Qwen_2_5M
from .Qwen_Qwen_2_5_Max              import Qwen_Qwen_2_5_Max
from .Qwen_Qwen_2_72B                import Qwen_Qwen_2_72B
from .Qwen_Qwen_3                    import Qwen_Qwen_3
from .StabilityAI_SD35Large          import StabilityAI_SD35Large

class HuggingSpace(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://huggingface.co/spaces"

    working = True

    default_model = Qwen_Qwen_2_72B.default_model
    default_image_model = BlackForestLabs_Flux1Dev.default_model
    default_vision_model = Microsoft_Phi_4_Multimodal.default_model
    providers = [
        BlackForestLabs_Flux1Dev,
        BlackForestLabs_Flux1KontextDev,
        CohereForAI_C4AI_Command,
        DeepseekAI_JanusPro7b,
        Microsoft_Phi_4_Multimodal,
        Qwen_Qwen_2_5,
        Qwen_Qwen_2_5M,
        Qwen_Qwen_2_5_Max,
        Qwen_Qwen_2_72B,
        Qwen_Qwen_3,
        StabilityAI_SD35Large,
    ]

    @classmethod
    def get_parameters(cls, **kwargs) -> dict:
        parameters = {}
        for provider in cls.providers:
            parameters = {**parameters, **provider.get_parameters(**kwargs)}
        return parameters

    @classmethod
    def get_models(cls, **kwargs) -> list[str]:
        if not cls.models:
            models = []
            image_models = []
            vision_models = []
            cls.model_aliases = {}
            for provider in cls.providers:
                models.extend(provider.get_models(**kwargs))
                models.extend(provider.model_aliases.keys())
                image_models.extend(provider.image_models)
                vision_models.extend(provider.vision_models)
                cls.model_aliases.update(provider.model_aliases)
            models = list(set(models))
            models.sort()
            cls.models = models
            cls.image_models = list(set(image_models))
            cls.vision_models = list(set(vision_models))
        return cls.models

    @classmethod
    async def create_async_generator(
        cls, model: str, messages: Messages, media: MediaListType = None, **kwargs
    ) -> AsyncResult:
        if not model and media is not None:
            model = cls.default_vision_model
        is_started = False
        random.shuffle(cls.providers)
        for provider in cls.providers:
            if model in provider.model_aliases or model in provider.get_models():
                alias = provider.model_aliases[model] if model in provider.model_aliases else model
                async for chunk in provider.create_async_generator(alias, messages, media=media, **kwargs):
                    is_started = True
                    yield chunk
            if is_started:
                return

for provider in HuggingSpace.providers:
    provider.parent = HuggingSpace.__name__
    provider.hf_space = True
