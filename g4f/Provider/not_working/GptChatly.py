from __future__ import annotations

from ...requests import Session, get_session_from_browser
from ...typing       import Messages
from ..base_provider import AsyncProvider


class GptChatly(AsyncProvider):
    url = "https://gptchatly.com"
    working = False
    supports_message_history = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        session: Session = None,
        **kwargs
    ) -> str:
        if not session:
            session = get_session_from_browser(cls.url, proxy=proxy, timeout=timeout)
        if model.startswith("gpt-4"):
            chat_url = f"{cls.url}/fetch-gpt4-response"
        else:
            chat_url = f"{cls.url}/felch-response"
        data = {
            "past_conversations": messages
        }
        response = session.post(chat_url, json=data)
        response.raise_for_status()
        return response.json()["chatGPTResponse"]