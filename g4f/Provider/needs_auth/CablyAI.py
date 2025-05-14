from __future__ import annotations

from ...typing import Messages, AsyncResult
from ..template import OpenaiTemplate

class CablyAI(OpenaiTemplate):
    url = "https://cablyai.com/chat"
    login_url = "https://cablyai.com"
    api_base = "https://cablyai.com/v1"

    working = True
    needs_auth = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        stream: bool = False,
        **kwargs
    ) -> AsyncResult:      
        headers = {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Referer": f"{cls.url}/chat",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }
        return super().create_async_generator(
            model=model,
            messages=messages,
            api_key=api_key,
            stream=stream,
            headers=headers,
            **kwargs
        )
