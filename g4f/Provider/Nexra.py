from __future__ import annotations

from aiohttp import ClientSession
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from .nexra.NexraBing import NexraBing
from .nexra.NexraChatGPT import NexraChatGPT
from .nexra.NexraChatGPT4o import NexraChatGPT4o
from .nexra.NexraChatGPTWeb import NexraChatGPTWeb
from .nexra.NexraGeminiPro import NexraGeminiPro
from .nexra.NexraImageURL import NexraImageURL
from .nexra.NexraLlama import NexraLlama
from .nexra.NexraQwen import NexraQwen

class Nexra(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://nexra.aryahcr.cc"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    default_model = 'gpt-3.5-turbo'
    image_model = 'sdxl-turbo'
    
    models = (
        *NexraBing.models,
        *NexraChatGPT.models,
        *NexraChatGPT4o.models,
        *NexraChatGPTWeb.models,
        *NexraGeminiPro.models,
        *NexraImageURL.models,
        *NexraLlama.models,
        *NexraQwen.models,
    )

    model_to_provider = {
        **{model: NexraChatGPT for model in NexraChatGPT.models},
        **{model: NexraChatGPT4o for model in NexraChatGPT4o.models},
        **{model: NexraChatGPTWeb for model in NexraChatGPTWeb.models},
        **{model: NexraGeminiPro for model in NexraGeminiPro.models},
        **{model: NexraImageURL for model in NexraImageURL.models},
        **{model: NexraLlama for model in NexraLlama.models},
        **{model: NexraQwen for model in NexraQwen.models},
        **{model: NexraBing for model in NexraBing.models},
    }
    
    model_aliases = {
        "gpt-4": "gpt-4-0613",
        "gpt-4": "gpt-4-32k",
        "gpt-4": "gpt-4-0314",
        "gpt-4": "gpt-4-32k-0314",
        
        "gpt-3.5-turbo": "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo": "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo": "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo": "gpt-3.5-turbo-0301",
        
        "gpt-3": "text-davinci-003",
        "gpt-3": "text-davinci-002",
        "gpt-3": "code-davinci-002",
        "gpt-3": "text-curie-001",
        "gpt-3": "text-babbage-001",
        "gpt-3": "text-ada-001",
        "gpt-3": "text-ada-001",
        "gpt-3": "davinci",
        "gpt-3": "curie",
        "gpt-3": "babbage",
        "gpt-3": "ada",
        "gpt-3": "babbage-002",
        "gpt-3": "davinci-002",
        
        "gpt-4": "gptweb",
        
        "gpt-4": "Bing (Balanced)",
        "gpt-4": "Bing (Creative)",
        "gpt-4": "Bing (Precise)",
        
        "dalle-2": "dalle2",
        "sdxl": "sdxl-turbo",
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
    def get_api_endpoint(cls, model: str) -> str:
        provider_class = cls.model_to_provider.get(model)

        if provider_class:
            return provider_class.api_endpoint
        raise ValueError(f"API endpoint for model {model} not found.")

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        api_endpoint = cls.get_api_endpoint(model)

        provider_class = cls.model_to_provider.get(model)

        if provider_class:
            async for response in provider_class.create_async_generator(model, messages, proxy, **kwargs):
                yield response
        else:
            raise ValueError(f"Provider for model {model} not found.")
