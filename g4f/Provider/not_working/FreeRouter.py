from __future__ import annotations

from ..template import OpenaiTemplate

class FreeRouter(OpenaiTemplate):
    label = "CablyAI FreeRouter"
    url = "https://freerouter.cablyai.com"
    api_base = "https://freerouter.cablyai.com/v1"
    working = False
