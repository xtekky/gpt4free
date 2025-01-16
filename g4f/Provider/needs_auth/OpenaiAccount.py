from __future__ import annotations

from .OpenaiChat import OpenaiChat

class OpenaiAccount(OpenaiChat):
    needs_auth = True
    parent = "OpenaiChat"
    image_models = ["dall-e-3",  "gpt-4", "gpt-4o"]
    default_model = "gpt-4o"
    default_vision_model = default_model
    default_image_model = "dall-e-3"
    fallback_models = [*OpenaiChat.fallback_models, default_image_model]