from __future__ import annotations

from ..template.OpenaiTemplate import OpenaiTemplate

class xAI(OpenaiTemplate):
    url = "https://console.x.ai"
    login_url = "https://console.x.ai"
    api_base = "https://api.x.ai/v1"
    working = True
    needs_auth = True
    models_needs_auth = True