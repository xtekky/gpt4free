from __future__ import annotations

import asyncio

try:
    from duckai import DuckAI
    has_requirements = True
except ImportError:
    has_requirements = False

from ...typing import CreateResult, Messages
from ..base_provider import AbstractProvider, ProviderModelMixin
from ..helper import get_last_user_message

class DuckDuckGo(AbstractProvider, ProviderModelMixin):
    label = "Duck.ai (duckduckgo_search)"
    url = "https://duckduckgo.com/aichat"
    api_base = "https://duckduckgo.com/duckchat/v1/"
    
    working = has_requirements
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "gpt-4o-mini"
    models = [default_model, "meta-llama/Llama-3.3-70B-Instruct-Turbo", "claude-3-haiku-20240307", "o3-mini", "mistralai/Mistral-Small-24B-Instruct-2501"]

    duck_ai: DuckAI = None

    model_aliases = {
        "gpt-4": "gpt-4o-mini",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "mixtral-small-24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    }

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 60,
        **kwargs
    ) -> CreateResult:
        if not has_requirements:
            raise ImportError("duckai is not installed. Install it with `pip install -U duckai`.")
        if cls.duck_ai is None:
            cls.duck_ai = DuckAI(proxy=proxy, timeout=timeout)
        model = cls.get_model(model)
        yield cls.duck_ai.chat(get_last_user_message(messages), model, timeout)