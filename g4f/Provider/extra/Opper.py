from __future__ import annotations

from ..template import OpenaiTemplate


class Opper(OpenaiTemplate):
    label = "Opper"
    url = "https://opper.ai"
    login_url = "https://platform.opper.ai"
    base_url = "https://api.opper.ai/v3/compat"
    working = True
    needs_auth = True
    models_needs_auth = True
    default_model = "claude-sonnet-4-6"
