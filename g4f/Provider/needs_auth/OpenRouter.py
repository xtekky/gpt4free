from __future__ import annotations

from ..template import OpenaiTemplate
from ...config import DEFAULT_MODEL

class OpenRouter(OpenaiTemplate):
    label = "OpenRouter"
    url = "https://openrouter.ai"
    login_url = "https://openrouter.ai/settings/keys"
    api_base = "https://openrouter.ai/api/v1"
    working = True
    needs_auth = True
    default_model = DEFAULT_MODEL
    active_by_default = True

class OpenRouterFree(OpenRouter):
    parent = "OpenRouter"
    label = "OpenRouter (free)"

    @classmethod
    def get_models(cls, api_key: str = None, **kwargs):
        models = super().get_models(api_key=api_key, **kwargs)
        models = [model for model in models if model.endswith(":free")]
        cls.model_aliases = {model.replace(":free", ""): model for model in models}
        return [model.replace(":free", "") for model in models]
    
    @classmethod
    def get_model(cls, model: str, **kwargs) -> str:
        # Load model aliases if not already done
        cls.get_models(**kwargs)
        # Map the model to its alias if it exists
        return super().get_model(model, **kwargs)