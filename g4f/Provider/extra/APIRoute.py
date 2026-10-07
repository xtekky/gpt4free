from __future__ import annotations

from ..template import OpenaiTemplate


class APIRoute(OpenaiTemplate):
    label = "API Route"
    url = "https://www.api-route.com"
    login_url = "https://www.api-route.com/api-keys"
    base_url = "https://global.api-route.com/v1"
    working = True
    needs_auth = True
    models_needs_auth = True
    default_model = "gpt-6.1-sol"
