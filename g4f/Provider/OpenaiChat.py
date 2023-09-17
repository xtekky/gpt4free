from __future__ import annotations

from curl_cffi.requests import AsyncSession
import uuid
import json

from .base_provider import AsyncProvider, get_cookies, format_prompt
from ..typing import AsyncGenerator


class OpenaiChat(AsyncProvider):
    url                   = "https://chat.openai.com"
    needs_auth            = True
    working               = True
    supports_gpt_35_turbo = True
    _access_token         = None

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        access_token: str = None,
        cookies: dict = None,
        **kwargs: dict
    ) -> AsyncGenerator:
        proxies = None
        if proxy:
            if "://" not in proxy:
                proxy = f"http://{proxy}"
            proxies = {
                "http": proxy,
                "https": proxy
            }
        if not access_token:
            access_token = await cls.get_access_token(cookies, proxies)
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {access_token}",
        }
        async with AsyncSession(proxies=proxies, headers=headers, impersonate="chrome107") as session:
            messages = [
                {
                    "id": str(uuid.uuid4()),
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": [format_prompt(messages)]},
                },
            ]
            data = {
                "action": "next",
                "messages": messages,
                "conversation_id": None,
                "parent_message_id": str(uuid.uuid4()),
                "model": "text-davinci-002-render-sha",
                "history_and_training_disabled": True,
            }
            response = await session.post("https://chat.openai.com/backend-api/conversation", json=data)
            response.raise_for_status()
            last_message = None
            for line in response.content.decode().splitlines():
                if line.startswith("data: "):
                    line = line[6:]
                    if line != "[DONE]":
                        line = json.loads(line)
                        if "message" in line:
                            last_message = line["message"]["content"]["parts"][0]
            return last_message


    @classmethod
    async def get_access_token(cls, cookies: dict = None, proxies: dict = None):
        if not cls._access_token:
            cookies = cookies if cookies else get_cookies("chat.openai.com")
            async with AsyncSession(proxies=proxies, cookies=cookies, impersonate="chrome107") as session:
                response = await session.get("https://chat.openai.com/api/auth/session")
                response.raise_for_status()
                cls._access_token = response.json()["accessToken"]
        return cls._access_token


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("proxy", "str"),
            ("access_token", "str"),
            ("cookies", "dict[str, str]")
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"