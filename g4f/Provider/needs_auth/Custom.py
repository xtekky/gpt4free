from __future__ import annotations

from .OpenaiAPI import OpenaiAPI

class Custom(OpenaiAPI):
    label = "Custom"
    url = None
    login_url = "http://localhost:8080"
    working = True
    api_base = "http://localhost:8080/v1"
    needs_auth = False