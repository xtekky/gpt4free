from __future__         import annotations

from ...typing           import Messages
from ..base_provider     import BaseProvider, CreateResult
from ...requests         import get_session_from_browser
from uuid               import uuid4

class Bestim(BaseProvider):
    url = "https://chatgpt.bestim.org"
    working = False
    supports_gpt_35_turbo = True
    supports_message_history = True
    supports_stream = True

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None, 
        **kwargs
    ) -> CreateResult:
        session = get_session_from_browser(cls.url, proxy=proxy)
        headers = {
            'Accept': 'application/json, text/event-stream',
        }
        data = {
            "messagesHistory": [{
                "id": str(uuid4()),
                "content": m["content"],
                "from": "you" if m["role"] == "user" else "bot"
            } for m in messages],
            "type": "chat",
        }
        response = session.post(
            url="https://chatgpt.bestim.org/chat/send2/",
            json=data,
            headers=headers,
            stream=True
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"event: trylimit"):
                yield line.decode().removeprefix("data: ")






            




