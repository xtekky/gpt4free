from __future__ import annotations

from ...template import OpenaiTemplate

class MiniMax(OpenaiTemplate):
    label = "MiniMax API"
    url = "https://www.hailuo.ai/chat"
    login_url = "https://intl.minimaxi.com/user-center/basic-information/interface-key"
    base_url = "https://api.minimaxi.chat/v1"
    working = True
    needs_auth = True

    default_model = "MiniMax-M2.7"
    default_vision_model = default_model
    models = [default_model, "MiniMax-M2.7-highspeed", "MiniMax-Text-01", "abab6.5s-chat"]
    model_aliases = {"MiniMax": default_model}
