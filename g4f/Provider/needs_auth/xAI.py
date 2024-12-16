from __future__ import annotations

from .OpenaiAPI import OpenaiAPI

class xAI(OpenaiAPI):
    label = "xAI"
    url = "https://console.x.ai"
    api_base = "https://api.x.ai/v1"
    working = True