from __future__ import annotations

from ..errors import ModelNotSupportedError
from .template import OpenaiTemplate

class CablyAI(OpenaiTemplate):
    url = "https://cablyai.com/chat"
    login_url = "https://cablyai.com"
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
        'deepseek-r1-uncensored',
        'deepseek-r1',
        'deepseek-reasoner',
        'deepseek-v3',
        'andy-3.5',
        'hermes-3-llama-3.2-3b',
        'llama-3.1-8b-instruct',
        'o3-mini',
        'o3-mini-low',
        'sonar-reasoning',
        'tinyswallow1.5b',
    ]
      
    model_aliases = {
        "gpt-4o-mini": "searchgpt (free)",
        "deepseek-r1": "deepseek-r1-uncensored (free)",
        "deepseek-r1": "deepseek-reasoner (free)",
        "hermes-3": "hermes-3-llama-3.2-3b (free)",
        "llama-3.1-8b": "llama-3.1-8b-instruct (free)",
        "o3-mini-low": "o3-mini-low (free)",
        "o3-mini": "o3-mini-low (free)",
        "o3-mini": "o3-mini (free)",
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
