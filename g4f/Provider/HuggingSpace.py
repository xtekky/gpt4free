from __future__ import annotations
from typing import Any, Dict, List, Type
import inspect

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

from .hf_space import *

class HuggingSpace(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://hf.space"

    working = True
    
    default_model = ""
    default_image_model = ""
    
    image_models: List[str] = []
    models: List[str] = []
    model_aliases: Dict[str, str] = {}
    api_endpoints: Dict[str, str] = {}
    providers: Dict[str, Type[AsyncGeneratorProvider]] = {}

    @classmethod
    def initialize(cls):
        # Automatically add models, aliases and api_endpoints from all providers in hf_space
        for name, obj in globals().items():
            if isinstance(obj, type) and issubclass(obj, AsyncGeneratorProvider) and obj != cls:
                cls.image_models.extend(getattr(obj, 'image_models', []))
                cls.models.extend(getattr(obj, 'models', []))
                cls.model_aliases.update(getattr(obj, 'model_aliases', {}))
                
                # Add api_endpoint
                api_endpoint = getattr(obj, 'api_endpoint', "")
                if api_endpoint:
                    for model in getattr(obj, 'models', []):
                        cls.api_endpoints[model] = api_endpoint
                        cls.providers[model] = obj
        
        # Remove duplicates
        cls.image_models = list(set(cls.image_models))
        cls.models = list(set(cls.models))

    @classmethod
    async def create_async_generator(cls, model: str, messages: Messages, **kwargs) -> AsyncResult:
        if not cls.models:
            cls.initialize()

        if model not in cls.models:
            raise ValueError(f"Unsupported model: {model}")

        provider = cls.providers.get(model)
        if provider is None:
            raise ValueError(f"No provider found for model: {model}")

        # Get the signature of the provider's create_async_generator method
        sig = inspect.signature(provider.create_async_generator)
        
        # Filter kwargs to include only parameters accepted by the provider's method
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        # Add model and messages to filtered_kwargs
        filtered_kwargs['model'] = model
        filtered_kwargs['messages'] = messages

        # Add api_endpoint if it exists
        if model in cls.api_endpoints:
            filtered_kwargs['api_endpoint'] = cls.api_endpoints[model]

        async for result in provider.create_async_generator(**filtered_kwargs):
            yield result

# Call initialize on import
HuggingSpace.initialize()
