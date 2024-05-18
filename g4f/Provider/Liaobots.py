from __future__ import annotations

import uuid

from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_connector
from ..requests import raise_for_status

models = {
    "gpt-4o": {
        "context": "8K",
        "id": "gpt-4o-free",
        "maxLength": 31200,
        "model": "ChatGPT",
        "name": "GPT-4o-free",
        "provider": "OpenAI",
        "tokenLimit": 7800,
    },
    "gpt-3.5-turbo": {
        "id": "gpt-3.5-turbo",
        "name": "GPT-3.5-Turbo",
        "maxLength": 48000,
        "tokenLimit": 14000,
        "context": "16K",
    },
    "gpt-4-turbo": {
        "id": "gpt-4-turbo-preview",
        "name": "GPT-4-Turbo",
        "maxLength": 260000,
        "tokenLimit": 126000,
        "context": "128K",
    },
    "gpt-4": {
        "id": "gpt-4-plus",
        "name": "GPT-4-Plus",
        "maxLength": 130000,
        "tokenLimit": 31000,
        "context": "32K",
    },
    "gpt-4-0613": {
        "id": "gpt-4-0613",
        "name": "GPT-4-0613",
        "maxLength": 60000,
        "tokenLimit": 15000,
        "context": "16K",
    },
    "gemini-pro": {
        "id": "gemini-pro",
        "name": "Gemini-Pro",
        "maxLength": 120000,
        "tokenLimit": 30000,
        "context": "32K",
    },
    "claude-3-opus-20240229": {
        "id": "claude-3-opus-20240229",
        "name": "Claude-3-Opus",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "claude-3-sonnet-20240229": {
        "id": "claude-3-sonnet-20240229",
        "name": "Claude-3-Sonnet",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "claude-2.1": {
        "id": "claude-2.1",
        "name": "Claude-2.1-200k",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "claude-2.0": {
        "id": "claude-2.0",
        "name": "Claude-2.0-100k",
        "maxLength": 400000,
        "tokenLimit": 100000,
        "context": "100K",
    },
    "claude-instant-1": {
        "id": "claude-instant-1",
        "name": "Claude-instant-1",
        "maxLength": 400000,
        "tokenLimit": 100000,
        "context": "100K",
    }
}


class Liaobots(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://liaobots.site"
    working = True
    supports_message_history = True
    supports_system_message = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    default_model = "gpt-3.5-turbo"
    models = list(models)
    model_aliases = {
        "claude-v2": "claude-2"
    }
    _auth_code = ""
    _cookie_jar = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        auth: str = None,
        proxy: str = None,
        connector: BaseConnector = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "authority": "liaobots.com",
            "content-type": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        }
        async with ClientSession(
            headers=headers,
            cookie_jar=cls._cookie_jar,
            connector=get_connector(connector, proxy, True)
        ) as session:
            data = {
                "conversationId": str(uuid.uuid4()),
                "model": models[cls.get_model(model)],
                "messages": messages,
                "key": "",
                "prompt": kwargs.get("system_message", "You are a helpful assistant."),
            }
            if not cls._auth_code:
                async with session.post(
                    "https://liaobots.work/recaptcha/api/login",
                    data={"token": "abcdefghijklmnopqrst"},
                    verify_ssl=False
                ) as response:
                    await raise_for_status(response)
            try:
                async with session.post(
                    "https://liaobots.work/api/user",
                    json={"authcode": cls._auth_code},
                    verify_ssl=False
                ) as response:
                    await raise_for_status(response)
                    cls._auth_code = (await response.json(content_type=None))["authCode"]
                    if not cls._auth_code:
                        raise RuntimeError("Empty auth code")
                    cls._cookie_jar = session.cookie_jar
                async with session.post(
                    "https://liaobots.work/api/chat",
                    json=data,
                    headers={"x-auth-code": cls._auth_code},
                    verify_ssl=False
                ) as response:
                    await raise_for_status(response)
                    async for chunk in response.content.iter_any():
                        if b"<html coupert-item=" in chunk:
                            raise RuntimeError("Invalid session")
                        if chunk:
                            yield chunk.decode(errors="ignore")
            except:
                async with session.post(
                    "https://liaobots.work/api/user",
                    json={"authcode": "pTIQr4FTnVRfr"},
                    verify_ssl=False
                ) as response:
                    await raise_for_status(response)
                    cls._auth_code = (await response.json(content_type=None))["authCode"]
                    if not cls._auth_code:
                        raise RuntimeError("Empty auth code")
                    cls._cookie_jar = session.cookie_jar
                async with session.post(
                    "https://liaobots.work/api/chat",
                    json=data,
                    headers={"x-auth-code": cls._auth_code},
                    verify_ssl=False
                ) as response:
                    await raise_for_status(response)
                    async for chunk in response.content.iter_any():
                        if b"<html coupert-item=" in chunk:
                            raise RuntimeError("Invalid session")
                        if chunk:
                            yield chunk.decode(errors="ignore")
