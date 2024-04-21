from __future__ import annotations

from .OpenaiChat import OpenaiChat

class OpenaiAccount(OpenaiChat):
    needs_auth = True
    parent = "OpenaiChat"
    image_models = ["dall-e"]