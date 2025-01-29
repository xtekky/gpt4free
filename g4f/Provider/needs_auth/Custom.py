from __future__ import annotations

from ..template import OpenaiTemplate

class Custom(OpenaiTemplate):
    label = "Custom Provider"
    working = True
    needs_auth = False
    api_base = "http://localhost:8080/v1"
    sort_models = False

class Feature(Custom):
    label = "Feature Provider"
    working = False