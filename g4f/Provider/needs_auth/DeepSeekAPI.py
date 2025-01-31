from __future__ import annotations

import os
import json
import time
from typing import AsyncIterator
import asyncio

from ..base_provider import AsyncAuthedProvider
from ...requests import get_args_from_nodriver
from ...providers.response import AuthResult, RequestLogin, Reasoning, JsonConversation, FinishReason
from ...typing import AsyncResult, Messages

try:
    from curl_cffi import requests
    from dsk.api import DeepSeekAPI, AuthenticationError, DeepSeekPOW

    class DeepSeekAPIArgs(DeepSeekAPI):
        def __init__(self, args: dict):
            args.pop("headers")
            self.auth_token = args.pop("api_key")
            if not self.auth_token or not isinstance(self.auth_token, str):
                raise AuthenticationError("Invalid auth token provided")
            self.args = args
            self.pow_solver = DeepSeekPOW()

        def _make_request(self, method: str, endpoint: str, json_data: dict, pow_required: bool = False):
            url = f"{self.BASE_URL}{endpoint}"
            headers = self._get_headers()
            if pow_required:
                challenge = self._get_pow_challenge()
                pow_response = self.pow_solver.solve_challenge(challenge)
                headers = self._get_headers(pow_response)

            response = requests.request(
                method=method,
                url=url,
                json=json_data, **{
                    "headers":headers,
                    "impersonate":'chrome',
                    "timeout":None,
                    **self.args
                }
            )
            return response.json()
except ImportError:
    pass

class DeepSeekAPI(AsyncAuthedProvider):
    url = "https://chat.deepseek.com"
    working = False
    needs_auth = True
    use_nodriver = True
    _access_token = None

    @classmethod
    async def on_auth_async(cls, proxy: str = None, **kwargs) -> AsyncIterator:
        yield RequestLogin(cls.__name__, os.environ.get("G4F_LOGIN_URL") or "")
        async def callback(page):
            while True:
                await asyncio.sleep(1)
                cls._access_token = json.loads(await page.evaluate("localStorage.getItem('userToken')") or "{}").get("value")
                if cls._access_token:
                    break
        args = await get_args_from_nodriver(cls.url, proxy, callback=callback)
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
        **kwargs
    ) -> AsyncResult:
        # Initialize with your auth token
        api = DeepSeekAPIArgs(auth_result.get_dict())

        # Create a new chat session
        if conversation is None:
            chat_id = api.create_chat_session()
            conversation = JsonConversation(chat_id=chat_id)

        is_thinking = 0
        for chunk in api.chat_completion(
            conversation.chat_id,
            messages[-1]["content"],
            thinking_enabled=True
        ):
            if chunk['type'] == 'thinking':
                if not is_thinking:
                    yield Reasoning(None, "Is thinking...")
                    is_thinking = time.time()
                yield Reasoning(chunk['content'])
            elif chunk['type'] == 'text':
                if is_thinking:
                    yield Reasoning(None, f"Thought for {time.time() - is_thinking:.2f}s")
                    is_thinking = 0
                yield chunk['content']
            if chunk['finish_reason']:
                yield FinishReason(chunk['finish_reason'])