from __future__ import annotations

import os
import json
from typing import AsyncIterator
from aiohttp import ClientSession, FormData

from ....typing import AsyncResult, Messages
from ...base_provider import AsyncAuthedProvider, ProviderModelMixin, format_prompt
from ..mini_max.crypt import CallbackResults, get_browser_callback, generate_yy_header, get_body_to_yy
from ....requests import get_args_from_nodriver, raise_for_status
from ....providers.response import AuthResult, JsonConversation, RequestLogin, TitleGeneration
from ...helper import get_last_user_message
from .... import debug

class Conversation(JsonConversation):
    def __init__(self, token: str, chatID: str, characterID: str = 1):
        self.token = token
        self.chatID = chatID
        self.characterID = characterID

class HailuoAI(AsyncAuthedProvider, ProviderModelMixin):
    label = "Hailuo AI"
    url = "https://www.hailuo.ai"
    working = True
    use_nodriver = True
    needs_auth = True
    supports_stream = True
    default_model = "minimax"

    @classmethod
    async def on_auth_async(cls, proxy: str = None, **kwargs) -> AsyncIterator:
        login_url = os.environ.get("G4F_LOGIN_URL")
        if login_url:
            yield RequestLogin(cls.label, login_url)
        callback_results = CallbackResults()
        yield AuthResult(
            **await get_args_from_nodriver(
                cls.url,
                proxy=proxy,
                callback=await get_browser_callback(callback_results)
            ),
            **callback_results.get_dict()
        )

    @classmethod
    async def create_authed(
        cls,
        model: str,
        messages: Messages,
        auth_result: AuthResult,
        return_conversation: bool = True,
        conversation: Conversation = None,
        **kwargs
    ) -> AsyncResult:
        args = auth_result.get_dict().copy()
        args.pop("impersonate")
        token = args.pop("token")
        path_and_query = args.pop("path_and_query")
        timestamp = args.pop("timestamp")

        async with ClientSession(**args) as session:
            if conversation is not None and conversation.token != token:
                conversation = None
            form_data = {
                "characterID": 1 if conversation is None else getattr(conversation, "characterID", 1),
                "msgContent": format_prompt(messages) if conversation is None else get_last_user_message(messages),
                "chatID": 0 if conversation is None else getattr(conversation, "chatID", 0),
                "searchMode": 0
            }
            data = FormData(default_to_multipart=True)
            for name, value in form_data.items():
                form_data[name] = str(value)
                data.add_field(name, str(value))
            headers = {
                "token": token,
                "yy": generate_yy_header(auth_result.path_and_query, get_body_to_yy(form_data), timestamp)
            }
            async with session.post(f"{cls.url}{path_and_query}", data=data, headers=headers) as response:
                await raise_for_status(response)
                event = None
                yield_content_len = 0
                async for line in response.content:
                    if not line:
                        continue
                    if line.startswith(b"event:"):
                        event = line[6:].decode(errors="replace").strip()
                        if event == "close_chunk":
                            break
                    if line.startswith(b"data:"):
                        try:
                            data = json.loads(line[5:])
                        except json.JSONDecodeError as e:
                            debug.log(f"Failed to decode JSON: {line}, error: {e}")
                            continue
                        if event == "send_result":
                            send_result = data["data"]["sendResult"]
                            if "chatTitle" in send_result:
                                yield TitleGeneration(send_result["chatTitle"])
                            if "chatID" in send_result and return_conversation:
                                yield Conversation(token, send_result["chatID"])
                        elif event == "message_result":
                            message_result = data["data"]["messageResult"]
                            if "content" in message_result:
                                yield message_result["content"][yield_content_len:]
                                yield_content_len = len(message_result["content"])
