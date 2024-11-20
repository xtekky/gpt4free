from __future__ import annotations

from ..base_provider import ProviderModelMixin
from ..Copilot import Copilot

class CopilotAccount(Copilot, ProviderModelMixin):
    needs_auth = True
    parent = "Copilot"
    default_model = "Copilot"
    default_vision_model = default_model
    models = [default_model]
    image_models = models