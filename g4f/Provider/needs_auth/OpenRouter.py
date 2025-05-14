from __future__ import annotations

from ..template import OpenaiTemplate

class OpenRouter(OpenaiTemplate):
    label = "OpenRouter"
    url = "https://openrouter.ai"
    login_url = "https://openrouter.ai/settings/keys"
    api_base = "https://openrouter.ai/api/v1"
    working = True
    needs_auth = True