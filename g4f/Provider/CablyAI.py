from __future__ import annotations

from ..errors import ModelNotSupportedError
from .template import OpenaiTemplate

class CablyAI(OpenaiTemplate):
    url = "https://cablyai.com"
    login_url = url
    api_base = "https://cablyai.com/v1"
    api_key = "sk-your-openai-api-key"

    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-4o-mini'
    fallback_models = [
        default_model,
        'searchgpt',
        'llama-3.1-8b-instruct',
        'deepseek-r1-uncensored',
        'deepseek-v3',
        'tinyswallow1.5b',
        'andy-3.5',
        'o3-mini-low',
    ]
      
    model_aliases = {
        "gpt-4o-mini": "searchgpt",
        "llama-3.1-8b": "llama-3.1-8b-instruct",
        "deepseek-r1": "deepseek-r1-uncensored",
        "o3-mini": "o3-mini-low",
    }
    
    @classmethod
    def get_models(cls, api_key: str = None, api_base: str = None) -> list[str]:
        models = super().get_models(api_key, api_base);
        return [f"{m} (free)" for m in models if m in cls.fallback_models] + models

    @classmethod
    def get_model(cls, model: str, **kwargs) -> str:
        try:
            model = super().get_model(model, **kwargs)
            return model.split(" (free)")[0]
        except ModelNotSupportedError:
            if f"f{model} (free)" in cls.models:
                return model
            raise

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        stream: bool = True,
        **kwargs
    ) -> AsyncResult:      
        api_key = api_key or cls.api_key
        headers = {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Referer": f"{cls.url}/chat",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }
        return super().create_async_generator(
            model=model,
            messages=messages,
            api_key=api_key,
            stream=stream,
            headers=headers,
            **kwargs
        )
