from __future__ import annotations

import json
from http.cookiejar import CookieJar
try:
    from curl_cffi.requests import Session, CurlWsFlag
    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False

from .base_provider import AbstractProvider, BaseConversation
from .helper import format_prompt
from ..typing import CreateResult, Messages
from ..errors import MissingRequirementsError
from ..requests.raise_for_status import raise_for_status
from .. import debug

class Conversation(BaseConversation):
    conversation_id: str
    cookie_jar: CookieJar

    def __init__(self, conversation_id: str, cookie_jar: CookieJar):
        self.conversation_id = conversation_id
        self.cookie_jar = cookie_jar

class Copilot(AbstractProvider):
    label = "Microsoft Copilot"
    url = "https://copilot.microsoft.com"
    working = True
    supports_stream = True

    websocket_url = "wss://copilot.microsoft.com/c/api/chat?api-version=2"
    conversation_url = f"{url}/c/api/conversations"

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        proxy: str = None,
        timeout: int = 900,
        conversation: Conversation = None,
        return_conversation: bool = False,
        **kwargs
    ) -> CreateResult:
        if not has_curl_cffi:
            raise MissingRequirementsError('Install or update "curl_cffi" package | pip install -U nodriver')

        cookies = conversation.cookie_jar if conversation is not None else None
        with Session(timeout=timeout, proxy=proxy, impersonate="chrome", cookies=cookies) as session:
            response = session.get(f"{cls.url}/")
            raise_for_status(response)
            if conversation is None:
                response = session.post(cls.conversation_url)
                raise_for_status(response)
                conversation_id = response.json().get("id")
                if return_conversation:
                    yield Conversation(conversation_id, session.cookies.jar)
                prompt = format_prompt(messages)
                if debug.logging:
                    print(f"Copilot: Created conversation: {conversation_id}")
            else:
                conversation_id = conversation.conversation_id
                prompt = messages[-1]["content"]
                if debug.logging:
                    print(f"Copilot: Use conversation: {conversation_id}")

            wss = session.ws_connect(cls.websocket_url)
            wss.send(json.dumps({
                "event": "send",
                "conversationId": conversation_id,
                "content": [{
                    "type": "text",
                    "text": prompt,
                }],
                "mode": "chat"
            }).encode(), CurlWsFlag.TEXT)
            while True:
                try:
                    msg = json.loads(wss.recv()[0])
                except:
                    break
                if msg.get("event") == "appendText":
                    yield msg.get("text")
                elif msg.get("event") in ["done", "partCompleted"]:
                    break