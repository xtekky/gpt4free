from __future__ import annotations

from ..typing import AsyncResult, Messages
from .template import OpenaiTemplate

class CablyAI(OpenaiTemplate):
    url = "https://cablyai.com"
    login_url = None
    needs_auth = False
    api_base = "https://cablyai.com/v1"
    working = True

    default_model = "Cably-80B"
    models = [default_model]
    model_aliases = {"cably-80b": default_model}

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        **kwargs
    ) -> AsyncResult:      
        headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Content-Type': 'application/json',
            'Origin': 'https://cablyai.com',
            'Referer': 'https://cablyai.com/chat',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }
        return super().create_async_generator(
            model=model,
            messages=messages,
            headers=headers,
            **kwargs
        )
