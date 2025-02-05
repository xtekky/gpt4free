from __future__ import annotations

from ..errors import ModelNotSupportedError
from .template import OpenaiTemplate

class CablyAI(OpenaiTemplate):
    label = "CablyAI"
    url = "https://cablyai.com"
    login_url = url
    api_base = "https://cablyai.com/v1"
    api_key = "sk-your-openai-api-key"
    
    working = True
    needs_auth = False
    
    default_model = 'gpt-4o-mini'
    reasoning_models = ['deepseek-r1-uncensored']
    fallback_models = [
        default_model,
        'searchgpt',
        'llama-3.1-8b-instruct',
        'deepseek-v3',
        'tinyswallow1.5b',
        'andy-3.5',
        'o3-mini-low',
    ] + reasoning_models
    
    model_aliases = {
        "searchgpt": "searchgpt (free)",
        "gpt-4o-mini": "searchgpt",
        "llama-3.1-8b": "llama-3.1-8b-instruct",
        "deepseek-r1": "deepseek-r1-uncensored",
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