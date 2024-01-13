from __future__ import annotations

from urllib.parse import unquote

from ...typing import AsyncResult, Messages
from ..base_provider import AbstractProvider
from ...webdriver import WebDriver
from ...requests import Session, get_session_from_browser

class AiChatting(AbstractProvider):
    url = "https://www.aichatting.net"
    supports_gpt_35_turbo = True
    _session: Session = None

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        timeout: int = 120,
        webdriver: WebDriver = None,
        **kwargs
    ) -> AsyncResult:
        if not cls._session:
            cls._session = get_session_from_browser(cls.url, webdriver, proxy, timeout)
        visitorId = unquote(cls._session.cookies.get("aichatting.website.visitorId"))
                        
        headers = {
            "accept": "application/json, text/plain, */*",
            "lang": "en",
            "source": "web"
        }
        data = {
            "roleId": 0,
        }
        try:
            response = cls._session.post("https://aga-api.aichatting.net/aigc/chat/record/conversation/create", json=data, headers=headers)
            response.raise_for_status()
            conversation_id = response.json()["data"]["conversationId"]
        except Exception as e:
            cls.reset()
            raise e
        headers = {
            "authority": "aga-api.aichatting.net",
            "accept": "text/event-stream,application/json, text/event-stream",
            "lang": "en",
            "source": "web",
            "vtoken": visitorId,
        }
        data = {
            "spaceHandle": True,
            "roleId": 0,
            "messages": messages,
            "conversationId": conversation_id,
        }
        response = cls._session.post("https://aga-api.aichatting.net/aigc/chat/v2/stream", json=data, headers=headers, stream=True)
        response.raise_for_status()
        for chunk in response.iter_lines():
            if chunk.startswith(b"data:"):
                yield chunk[5:].decode().replace("-=- --", " ").replace("-=-n--", "\n").replace("--@DONE@--", "")

    @classmethod
    def reset(cls):
        cls._session = None