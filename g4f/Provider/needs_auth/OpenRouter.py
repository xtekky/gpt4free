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
    active_by_default = True
    default_model = DEFAULT_MODEL