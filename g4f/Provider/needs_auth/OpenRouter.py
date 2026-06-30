from __future__ import annotations

from ..template import OpenaiTemplate

class OpenRouter(OpenaiTemplate):
    label = "OpenRouter"
    url = "https://openrouter.ai"
    login_url = "https://openrouter.ai/settings/keys"
    base_url = "https://openrouter.ai/api/v1"
    working = True
    needs_auth = True
    default_model = "openrouter/auto"

class OpenRouterFree(OpenaiTemplate):
    label = "OpenRouter (free)"
    url = "https://openrouter.ai"
    login_url = "https://openrouter.ai/settings/keys"
    base_url = "https://openrouter.ai/api/v1"
    backup_url = "https://g4f.space/api/openrouter"
    max_tokens = 8192
    working = True
    active_by_default = True
    default_model = "openrouter/free"

    @classmethod
    def get_models(cls, api_key: str = None, **kwargs):
        models = super().get_models(api_key=api_key, **kwargs)
        models = [model for model in models if model.endswith(":free") or model.endswith("/free")]
        cls.model_aliases = {model.replace(":free", ""): model for model in models}
        return models