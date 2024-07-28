from __future__ import annotations

import uuid

from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_connector
from ..requests import raise_for_status

models = {
    "gpt-4o-mini-free": {
        "id": "gpt-4o-mini-free",
        "name": "GPT-4o-Mini-Free",
        "model": "ChatGPT",
        "provider": "OpenAI",
        "maxLength": 31200,
        "tokenLimit": 7800,
        "context": "8K",
    },
    "gpt-4o-mini": {
        "id": "gpt-4o-mini",
        "name": "GPT-4o-Mini",
        "model": "ChatGPT",
        "provider": "OpenAI",
        "maxLength": 260000,
        "tokenLimit": 126000,
        "context": "128K",
    },
    "gpt-4o-free": {
        "context": "8K",
        "id": "gpt-4o-free",
        "maxLength": 31200,
        "model": "ChatGPT",
        "name": "GPT-4o-free",
        "provider": "OpenAI",
        "tokenLimit": 7800,
    },
    "gpt-4-turbo-2024-04-09": {
        "id": "gpt-4-turbo-2024-04-09",
        "name": "GPT-4-Turbo",
        "model": "ChatGPT",
        "provider": "OpenAI",
        "maxLength": 260000,
        "tokenLimit": 126000,
        "context": "128K",
    },
    "gpt-4o": {
        "context": "128K",
        "id": "gpt-4o",
        "maxLength": 124000,
        "model": "ChatGPT",
        "name": "GPT-4o",
        "provider": "OpenAI",
        "tokenLimit": 62000,
    },
    "gpt-4-0613": {
        "id": "gpt-4-0613",
        "name": "GPT-4",
        "model": "ChatGPT",
        "provider": "OpenAI",
        "maxLength": 260000,
        "tokenLimit": 126000,
        "context": "128K",
    },
    "gpt-4-turbo": {
        "id": "gpt-4-turbo",
        "name": "GPT-4-Turbo",
        "model": "ChatGPT",
        "provider": "OpenAI",
        "maxLength": 260000,
        "tokenLimit": 126000,
        "context": "128K",
    },
}



class Liaobots(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://liaobots.site"
    working = True
    supports_message_history = True
    supports_system_message = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    default_model = "gpt-4o"
    models = list(models.keys())
    model_aliases = {
        "gpt-4o-mini": "gpt-4o-mini-free",
        "gpt-4o": "gpt-4o-free",
        "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
        "gpt-4-": "gpt-4-0613",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-opus": "claude-3-opus-20240229-aws",
        "claude-3-opus": "claude-3-opus-20240229-gcp",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "gemini-pro": "gemini-1.5-pro-latest",
        "gemini-pro": "gemini-1.0-pro-latest",
        "gemini-flash": "gemini-1.5-flash-latest",
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
                "model": models[model],
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

    @classmethod
    def get_model(cls, model: str) -> str:
        """
        Retrieve the internal model identifier based on the provided model name or alias.
        """
        if model in cls.model_aliases:
            model = cls.model_aliases[model]
        if model not in models:
            raise ValueError(f"Model '{model}' is not supported.")
        return model

    @classmethod
    def is_supported(cls, model: str) -> bool:
        """
        Check if the given model is supported.
        """
        return model in models or model in cls.model_aliases

    @classmethod
    async def initialize_auth_code(cls, session: ClientSession) -> None:
        """
        Initialize the auth code by making the necessary login requests.
        """
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

    @classmethod
    async def ensure_auth_code(cls, session: ClientSession) -> None:
        """
        Ensure the auth code is initialized, and if not, perform the initialization.
        """
        if not cls._auth_code:
            await cls.initialize_auth_code(session)
