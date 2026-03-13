from __future__ import annotations

from ...template import OpenaiTemplate

class MiniMax(OpenaiTemplate):
    label = "MiniMax API"
    url = "https://www.minimax.io"
    login_url = "https://platform.minimax.io/user-center/basic-information/interface-key"
    base_url = "https://api.minimax.io/v1"
    working = True
    needs_auth = True

    default_model = "MiniMax-M2.5"
    default_vision_model = default_model
    models = [default_model, "MiniMax-M2.5-highspeed", "MiniMax-Text-01", "abab6.5s-chat"]
    model_aliases = {"MiniMax": default_model, "minimax": default_model}
