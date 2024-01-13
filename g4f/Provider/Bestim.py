from __future__         import annotations

from ..typing           import Messages
from .base_provider     import BaseProvider, CreateResult
from ..requests         import get_session_from_browser
from uuid               import uuid4
import requests

class Bestim(BaseProvider):
    url = "https://chatgpt.bestim.org"
    supports_gpt_35_turbo = True
    supports_message_history = True
    working = False
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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Accept': 'application/json, text/event-stream',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://chatgpt.bestim.org/chat/',
            'Origin': 'https://chatgpt.bestim.org',
            'Alt-Used': 'chatgpt.bestim.org',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'TE': 'trailers'
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
            headers=headers,
            json=data,
            proxies={"https": proxy},
            stream=True
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"event: trylimit"):
                yield line.decode().removeprefix("data: ")






            




