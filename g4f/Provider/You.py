from __future__ import annotations

import json
import base64
import uuid

from ..requests import StreamSession
from ..typing import AsyncGenerator, Messages
from .base_provider import AsyncGeneratorProvider, format_prompt


class You(AsyncGeneratorProvider):
    url = "https://you.com"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    _session_used = 0
    _session_token = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        **kwargs,
    ) -> AsyncGenerator:
        async with StreamSession(proxies={"https": proxy}, impersonate="chrome107", timeout=timeout) as session:
            headers = {
                "Accept": "text/event-stream",
                "Referer": f"{cls.url}/search?fromSearchBar=true&tbm=youchat",
            }
            data = {
                "q": format_prompt(messages),
                "domain": "youchat",
                "chat": "", "selectedChatMode": "gpt-4" if model == "gpt-4" else "default"
            }
            async with session.get(
                f"{cls.url}/api/streamingSearch",
                params=data,
                headers=headers,
                cookies=cls.get_cookies(await cls.get_session_token(proxy, timeout)) if model == "gpt-4" else None
            ) as response:
                response.raise_for_status()
                start = b'data: {"youChatToken": '
                async for line in response.iter_lines():
                    if line.startswith(start):
                        yield json.loads(line[len(start):-1])

    @classmethod
    async def get_session_token(cls, proxy: str, timeout: int):
        if not cls._session_token or cls._session_used >= 5:
            cls._session_token = await cls.create_session_token(proxy, timeout)
        cls._session_used += 1
        return cls._session_token

    def get_cookies(access_token: str, session_jwt: str = "0"):
        return {
            'stytch_session_jwt': session_jwt,
            'ydc_stytch_session': access_token,
            'ydc_stytch_session_jwt': session_jwt
        }

    @classmethod
    def get_jwt(cls):
        return base64.standard_b64encode(json.dumps({
            "event_id":f"event-id-{str(uuid.uuid4())}",
            "app_session_id":f"app-session-id-{str(uuid.uuid4())}",
            "persistent_id":f"persistent-id-{uuid.uuid4()}",
            "client_sent_at":"","timezone":"",
            "stytch_user_id":f"user-live-{uuid.uuid4()}",
            "stytch_session_id":f"session-live-{uuid.uuid4()}",
            "app":{"identifier":"you.com"},
            "sdk":{"identifier":"Stytch.js Javascript SDK","version":"3.3.0"
        }}).encode()).decode()

    @classmethod
    async def create_session_token(cls, proxy: str, timeout: int):
        async with StreamSession(proxies={"https": proxy}, impersonate="chrome110", timeout=timeout) as session:
            user_uuid = str(uuid.uuid4())
            auth_uuid = "507a52ad-7e69-496b-aee0-1c9863c7c8"
            auth_token = f"public-token-live-{auth_uuid}bb:public-token-live-{auth_uuid}19"
            auth = base64.standard_b64encode(auth_token.encode()).decode()
            async with session.post(
                "https://web.stytch.com/sdk/v1/passwords",
                headers={
                    "Authorization": f"Basic {auth}",
                    "X-SDK-Client": cls.get_jwt(),
                    "X-SDK-Parent-Host": "https://you.com"
                },
                json={
                    "email": f"{user_uuid}@gmail.com",
                    "password": f"{user_uuid}#{user_uuid}",
                    "session_duration_minutes": 129600
                }
            ) as response:
                if not response.ok:
                    raise RuntimeError(f"Response: {await response.text()}")
                return (await response.json())["data"]["session_token"]