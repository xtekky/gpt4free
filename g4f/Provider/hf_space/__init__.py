from __future__ import annotations

from ...typing import AsyncResult, Messages, ImagesType
from ...errors import ResponseError
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin

from .BlackForestLabsFlux1Dev        import BlackForestLabsFlux1Dev
from .BlackForestLabsFlux1Schnell    import BlackForestLabsFlux1Schnell
from .VoodoohopFlux1Schnell          import VoodoohopFlux1Schnell
from .StableDiffusion35Large         import StableDiffusion35Large
from .CohereForAI                    import CohereForAI
from .Qwen_QVQ_72B                   import Qwen_QVQ_72B
from .Qwen_Qwen_2_72B_Instruct       import Qwen_Qwen_2_72B_Instruct

class HuggingSpace(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://huggingface.co/spaces"
    parent = "HuggingFace"
    
    working = True
    
    default_model = BlackForestLabsFlux1Dev.default_model
    default_vision_model = Qwen_QVQ_72B.default_model
    providers = [BlackForestLabsFlux1Dev, BlackForestLabsFlux1Schnell, VoodoohopFlux1Schnell, StableDiffusion35Large, CohereForAI, Qwen_QVQ_72B, Qwen_Qwen_2_72B_Instruct]

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
            for provider in cls.providers:
                models.extend(provider.get_models(**kwargs))
                models.extend(provider.model_aliases.keys())
            models = list(set(models))
            models.sort()
            cls.models = models
        return cls.models

    @classmethod
    async def create_async_generator(
        cls, model: str, messages: Messages, images: ImagesType = None, **kwargs
    ) -> AsyncResult:
        if not model and images is not None:
            model = cls.default_vision_model
        is_started = False
        for provider in cls.providers:
            if model in provider.model_aliases:
                async for chunk in provider.create_async_generator(provider.model_aliases[model], messages, **kwargs):
                    is_started = True
                    yield chunk
            if is_started:
                return
        error = None
        for provider in cls.providers:
            if model in provider.get_models():
                try:
                    async for chunk in provider.create_async_generator(model, messages, **kwargs):
                        is_started = True
                        yield chunk
                    if is_started:
                        break
                except ResponseError as e:
                    if is_started:
                        raise e
                    error = e
        if not is_started and error is not None:
            raise error
