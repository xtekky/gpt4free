from __future__ import annotations

import os
import json
import time
from typing import AsyncIterator
import asyncio

from ..base_provider import AsyncAuthedProvider, ProviderModelMixin
from ...providers.helper import get_last_user_message
from ...requests import get_args_from_nodriver, get_nodriver
from ...providers.response import AuthResult, RequestLogin, Reasoning, JsonConversation, FinishReason
from ...typing import AsyncResult, Messages
try:
    from dsk.api import DeepSeekAPI as DskAPI
    has_dsk = True
except ImportError:
    has_dsk = False

class DeepSeekAPI(AsyncAuthedProvider, ProviderModelMixin):
    label = "DeepSeek"
    url = "https://chat.deepseek.com"
    working = has_dsk
    active_by_default = False
    needs_auth = True
    use_nodriver = True
    _access_token = None

    default_model = "deepseek-v3"
    models = ["deepseek-v3", "deepseek-r1"]
    model_aliases = {"deepseek-chat": "deepseek-v3"}

    @classmethod
    async def on_auth_async(cls, proxy: str = None, **kwargs) -> AsyncIterator:
        if not hasattr(cls, "browser"):
            cls.browser, cls.stop_browser = await get_nodriver()
        yield RequestLogin(cls.__name__, os.environ.get("G4F_LOGIN_URL") or "")
        async def callback(page):
            while True:
                await asyncio.sleep(1)
                cls._access_token = json.loads(await page.evaluate("localStorage.getItem('userToken')") or "{}").get("value")
                if cls._access_token:
                    break
        args = await get_args_from_nodriver(cls.url, proxy, callback=callback, browser=cls.browser)
        yield AuthResult(
            api_key=cls._access_token,
            **args
        )

    @classmethod
    async def create_authed(
        cls,
        model: str,
        messages: Messages,
        auth_result: AuthResult,
        conversation: JsonConversation = None,
        web_search: bool = False,
        **kwargs
    ) -> AsyncResult:
        # Initialize with your auth token
        api = DskAPI(auth_result.get_dict())

        # Create a new chat session
        if conversation is None:
            chat_id = api.create_chat_session()
            conversation = JsonConversation(chat_id=chat_id)

        is_thinking = 0
        for chunk in api.chat_completion(
            conversation.chat_id,
            get_last_user_message(messages),
            thinking_enabled=bool(model) and "deepseek-r1" in model,
            search_enabled=web_search,
            parent_message_id=getattr(conversation, "parent_id", None)
        ):
            if chunk['type'] == 'thinking':
                if not is_thinking:
                    yield Reasoning(status="Is thinking...")
                    is_thinking = time.time()
                yield Reasoning(chunk['content'])
            elif chunk['type'] == 'text':
                if is_thinking:
                    yield Reasoning(status=f"Thought for {time.time() - is_thinking:.2f}s")
                    is_thinking = 0
                if chunk['content']:
                    yield chunk['content']
            if 'message_id' in chunk:
                conversation.parent_id = chunk['message_id']
            if chunk['finish_reason']:
                if 'message_id' in chunk:
                    conversation.parent_id = chunk['message_id']
                    yield conversation
                yield FinishReason(chunk['finish_reason'])