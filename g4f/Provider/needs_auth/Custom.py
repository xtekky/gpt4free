from __future__ import annotations

from .OpenaiTemplate import OpenaiTemplate

class Custom(OpenaiTemplate):
    label = "Custom Provider"
    working = True
    api_base = "http://localhost:8080/v1"
    sort_models = False