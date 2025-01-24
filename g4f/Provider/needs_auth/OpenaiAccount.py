from __future__ import annotations

from .OpenaiChat import OpenaiChat

class OpenaiAccount(OpenaiChat):
    needs_auth = True
    parent = "OpenaiChat"
    default_model = "gpt-4o"
    default_vision_model = default_model
    default_image_model = OpenaiChat.default_image_model
    image_models = [default_model, default_image_model, "gpt-4"]
    fallback_models = [*OpenaiChat.fallback_models, default_image_model]
