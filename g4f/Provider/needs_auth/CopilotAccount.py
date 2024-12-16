from __future__ import annotations

from ..Copilot import Copilot

class CopilotAccount(Copilot):
    needs_auth = True
    parent = "Copilot"
    default_model = "Copilot"
    default_vision_model = default_model
    models = [default_model]
    image_models = models
    model_aliases = {
        "dall-e-3": default_model
    }