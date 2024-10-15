from __future__ import annotations

import uuid
from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_connector
from ..requests import raise_for_status

models = {
    "gpt-3.5-turbo": {
        "id": "gpt-3.5-turbo",
        "name": "GPT-3.5-Turbo",
        "model": "ChatGPT",
        "provider": "OpenAI",
        "maxLength": 48000,
        "tokenLimit": 14000,
        "context": "16K",
    },
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
        "id": "gpt-4o-free",
        "name": "GPT-4o-free",
        "model": "ChatGPT",
        "provider": "OpenAI",
        "maxLength": 31200,
        "tokenLimit": 7800,
        "context": "8K",
    },
    "gpt-4o-2024-08-06": {
        "id": "gpt-4o-2024-08-06",
        "name": "GPT-4o",
        "model": "ChatGPT",
        "provider": "OpenAI",
        "maxLength": 260000,
        "tokenLimit": 126000,
        "context": "128K",
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
    "grok-2": {
        "id": "grok-2",
        "name": "Grok-2",
        "model": "Grok",
        "provider": "x.ai",
        "maxLength": 400000,
        "tokenLimit": 100000,
        "context": "100K",
    },
    "grok-2-mini": {
        "id": "grok-2-mini",
        "name": "Grok-2-mini",
        "model": "Grok",
        "provider": "x.ai",
        "maxLength": 400000,
        "tokenLimit": 100000,
        "context": "100K",
    },
    "claude-3-opus-20240229": {
        "id": "claude-3-opus-20240229",
        "name": "Claude-3-Opus",
        "model": "Claude",
        "provider": "Anthropic",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "claude-3-opus-20240229-aws": {
        "id": "claude-3-opus-20240229-aws",
        "name": "Claude-3-Opus-Aws",
        "model": "Claude",
        "provider": "Anthropic",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "claude-3-opus-20240229-gcp": {
        "id": "claude-3-opus-20240229-gcp",
        "name": "Claude-3-Opus-Gcp",
        "model": "Claude",
        "provider": "Anthropic",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "claude-3-5-sonnet-20240620": {
        "id": "claude-3-5-sonnet-20240620",
        "name": "Claude-3.5-Sonnet",
        "model": "Claude",
        "provider": "Anthropic",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "claude-3-sonnet-20240229": {
        "id": "claude-3-sonnet-20240229",
        "name": "Claude-3-Sonnet",
        "model": "Claude",
        "provider": "Anthropic",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "claude-3-haiku-20240307": {
        "id": "claude-3-haiku-20240307",
        "name": "Claude-3-Haiku",
        "model": "Claude",
        "provider": "Anthropic",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "claude-2.1": {
        "id": "claude-2.1",
        "name": "Claude-2.1-200k",
        "model": "Claude",
        "provider": "Anthropic",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "gemini-1.5-flash-002": {
        "id": "gemini-1.5-flash-002",
        "name": "Gemini-1.5-Flash-1M",
        "model": "Gemini",
        "provider": "Google",
        "maxLength": 4000000,
        "tokenLimit": 1000000,
        "context": "1024K",
    },
    "gemini-1.5-pro-002": {
        "id": "gemini-1.5-pro-002",
        "name": "Gemini-1.5-Pro-1M",
        "model": "Gemini",
        "provider": "Google",
        "maxLength": 4000000,
        "tokenLimit": 1000000,
        "context": "1024K",
    },
}


class Liaobots(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://liaobots.site"
    working = True
    supports_message_history = True
    supports_system_message = True
    supports_gpt_4 = True
    default_model = "gpt-3.5-turbo"
    models = list(models.keys())
    
    model_aliases = {
        "gpt-4o-mini": "gpt-4o-mini-free",
        "gpt-4o": "gpt-4o-free",
        "gpt-4o": "gpt-4o-2024-08-06",
              
        "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
        "gpt-4": "gpt-4-0613",
        
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-opus": "claude-3-opus-20240229-aws",
        "claude-3-opus": "claude-3-opus-20240229-gcp",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-2.1": "claude-2.1",
        
        "gemini-flash": "gemini-1.5-flash-002",
        "gemini-pro": "gemini-1.5-pro-002",
    }
    
    _auth_code = ""
    _cookie_jar = None

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
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        auth: str = None,
        proxy: str = None,
        connector: BaseConnector = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
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
