from __future__ import annotations

import os
from typing import Optional

from ..providers.types import ProviderType
from .. import debug

class AuthManager:
    """Handles API key management"""
    aliases = {
        "GeminiPro": "Gemini",
        "PollinationsAI": "Pollinations",
        "OpenaiAPI": "Openai",
        "PuterJS": "Puter",
    }

    @classmethod
    def load_api_key(cls, provider: ProviderType) -> Optional[str]:
        """Load API key from config file"""
        if not provider.needs_auth and not hasattr(provider, "login_url"):
            return None
        provider_name = provider.get_parent()
        env_var = f"{provider_name.upper()}_API_KEY"
        api_key = os.environ.get(env_var)
        if not api_key and provider_name in cls.aliases:
            env_var = f"{cls.aliases[provider_name].upper()}_API_KEY"
            api_key = os.environ.get(env_var)
        if api_key:
            debug.log(f"Loading API key for {provider_name} from environment variable {env_var}")
            return api_key
        return None
