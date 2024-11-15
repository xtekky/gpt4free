from __future__ import annotations
from typing import Any, Dict
import inspect

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from .airforce.AirforceChat import AirforceChat
from .airforce.AirforceImage import AirforceImage

class Airforce(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://api.airforce"
    api_endpoint_completions = AirforceChat.api_endpoint
    api_endpoint_imagine2 = AirforceImage.api_endpoint
    working = True
    supports_stream = AirforceChat.supports_stream
    supports_system_message = AirforceChat.supports_system_message
    supports_message_history = AirforceChat.supports_message_history 
    
    default_model = AirforceChat.default_model
    models = [*AirforceChat.models, *AirforceImage.models]
    
    model_aliases = {
        **AirforceChat.model_aliases,
        **AirforceImage.model_aliases
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            return cls.default_model

    @classmethod
    async def create_async_generator(cls, model: str, messages: Messages, **kwargs) -> AsyncResult:
        model = cls.get_model(model)
        
        provider = AirforceChat if model in AirforceChat.text_models else AirforceImage

        if model not in provider.models:
            raise ValueError(f"Unsupported model: {model}")

        # Get the signature of the provider's create_async_generator method
        sig = inspect.signature(provider.create_async_generator)
        
        # Filter kwargs to only include parameters that the provider's method accepts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        # Add model and messages to filtered_kwargs
        filtered_kwargs['model'] = model
        filtered_kwargs['messages'] = messages

        async for result in provider.create_async_generator(**filtered_kwargs):
            yield result
