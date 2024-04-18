from __future__ import annotations

from .OpenaiChat import OpenaiChat

class OpenaiAccount(OpenaiChat):
    label = "OpenAI ChatGPT with Account"
    needs_auth = True