from __future__ import annotations

from ..template import OpenaiTemplate

class OpenaiAPI(OpenaiTemplate):
    label = "OpenAI API"
    url = "https://platform.openai.com"
    login_url = "https://platform.openai.com/settings/organization/api-keys"
    api_base = "https://api.openai.com/v1"
    working = True
    needs_auth = True