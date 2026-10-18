from __future__ import annotations

from ..template import OpenaiTemplate


class Onomeo(OpenaiTemplate):
    label = "onomeo"
    url = "https://onomeo.com"
    login_url = "https://onomeo.com/dashboard"
    base_url = "https://onomeo.com/v1"
    working = True
    needs_auth = True
    models_needs_auth = True
    default_model = "auto"
