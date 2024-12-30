from __future__ import annotations

from ...typing import AsyncResult, Messages
from ...errors import ResponseError
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin

from .BlackForestLabsFlux1Dev        import BlackForestLabsFlux1Dev
from .BlackForestLabsFlux1Schnell    import BlackForestLabsFlux1Schnell
from .VoodoohopFlux1Schnell          import VoodoohopFlux1Schnell

class HuggingSpace(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://huggingface.co/spaces"
    working = True
    default_model = BlackForestLabsFlux1Dev.default_model
    providers = [BlackForestLabsFlux1Dev, BlackForestLabsFlux1Schnell, VoodoohopFlux1Schnell]

    @classmethod
    def get_parameters(cls, **kwargs) -> dict:
        parameters = {}
        for provider in cls.providers:
            parameters = {**parameters, **provider.get_parameters(**kwargs)}
        return parameters

    @classmethod
    def get_models(cls, **kwargs) -> list[str]:
        if not cls.models:
            for provider in cls.providers:
                cls.models.extend(provider.get_models(**kwargs))
                cls.models.extend(provider.model_aliases.keys())
            cls.models = list(set(cls.models))
            cls.models.sort()
        return cls.models

    @classmethod
    async def create_async_generator(
        cls, model: str, messages: Messages, **kwargs
    ) -> AsyncResult:
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