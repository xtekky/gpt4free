from __future__ import annotations

from ..Copilot import Copilot

class CopilotAccount(Copilot):
    needs_auth = True
    use_nodriver = True
    parent = "Copilot"
    default_model = "Copilot"
    default_vision_model = default_model
    model_aliases = {
        "gpt-4": default_model,
        "gpt-4o": default_model,
        "o1": "Think Deeper",
        "dall-e-3": default_model
    }