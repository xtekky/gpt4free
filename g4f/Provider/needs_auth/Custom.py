from __future__ import annotations

from .OpenaiAPI import OpenaiAPI

class Custom(OpenaiAPI):
    label = "Custom Provider"
    url = None
    login_url = None
    working = True
    api_base = "http://localhost:8080/v1"
    needs_auth = False
    sort_models = False