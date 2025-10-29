from __future__ import annotations

import secrets
import string

from .template import OpenaiTemplate

class StringableInference(OpenaiTemplate):
    label = "Stringable Inference"
    url = "https://stringable-inference.onrender.com"
    api_base = "https://stringableinf.com/api"
    api_endpoint = "https://stringableinf.com/api/v1/chat/completions"

    working = False
    active_by_default = True
    default_model = "deepseek-v3.2"
    default_vision_model = "gpt-oss-120b"

    @classmethod
    def get_headers(cls, stream: bool, api_key: str = None, headers: dict = None) -> dict:
        return {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://g4f.dev/",
            "X-Title": "G4F Python",
            **(
                {"Authorization": f"Bearer {api_key}"}
                if api_key else {}
            ),
            **({} if headers is None else headers)
        }