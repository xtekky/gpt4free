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
    "gpt-4o-2024-11-20": {
        "id": "gpt-4o-2024-11-20",
        "name": "GPT-4o",
        "model": "ChatGPT",
        "provider": "OpenAI",
        "maxLength": 260000,
        "tokenLimit": 126000,
        "context": "128K",
    },
    "gpt-4o-mini-2024-07-18": {
        "id": "gpt-4o-mini-2024-07-18",
        "name": "GPT-4o-Mini",
        "model": "ChatGPT",
        "provider": "OpenAI",
        "maxLength": 260000,
        "tokenLimit": 126000,
        "context": "128K",
    },
    "o1-preview-2024-09-12": {
        "id": "o1-preview-2024-09-12",
        "name": "o1-preview",
        "model": "o1",
        "provider": "OpenAI",
        "maxLength": 400000,
        "tokenLimit": 100000,
        "context": "128K",
    },
    "o1-mini-2024-09-12": {
        "id": "o1-mini-2024-09-12",
        "name": "o1-mini",
        "model": "o1",
        "provider": "OpenAI",
        "maxLength": 400000,
        "tokenLimit": 100000,
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
    "claude-3-opus-20240229": {
        "id": "claude-3-opus-20240229",
        "name": "Claude-3-Opus",
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
    "claude-3-5-sonnet-20241022": {
        "id": "claude-3-5-sonnet-20241022",
        "name": "Claude-3.5-Sonnet-V2",
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
    "claude-3-opus-20240229-t": {
        "id": "claude-3-opus-20240229-t",
        "name": "Claude-3-Opus-T",
        "model": "Claude",
        "provider": "Anthropic",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "claude-3-5-sonnet-20241022-t": {
        "id": "claude-3-5-sonnet-20241022-t",
        "name": "Claude-3.5-Sonnet-V2-T",
        "model": "Claude",
        "provider": "Anthropic",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "gemini-2.0-flash-exp": {
        "id": "gemini-2.0-flash-exp",
        "name": "Gemini-2.0-Flash-Exp",
        "model": "Gemini",
        "provider": "Google",
        "maxLength": 4000000,
        "tokenLimit": 1000000,
        "context": "1024K",
    },
    "gemini-2.0-flash-thinking-exp": {
        "id": "gemini-2.0-flash-thinking-exp",
        "name": "Gemini-2.0-Flash-Thinking-Exp",
        "model": "Gemini",
        "provider": "Google",
        "maxLength": 4000000,
        "tokenLimit": 1000000,
        "context": "1024K",
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
    
    default_model = "gpt-4o-2024-11-20"
    models = list(models.keys())
    model_aliases = {
        "gpt-4o-mini": "gpt-4o-mini-free",
        "gpt-4o": default_model,
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-4": default_model,
        
        "o1-preview": "o1-preview-2024-09-12",
        "o1-mini": "o1-mini-2024-09-12",
        
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-opus": "claude-3-opus-20240229-t",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022-t",
        
        "gemini-2.0-flash": "gemini-2.0-flash-exp",
        "gemini-2.0-flash-thinking": "gemini-2.0-flash-thinking-exp",
        "gemini-1.5-flash": "gemini-1.5-flash-002",
        "gemini-1.5-pro": "gemini-1.5-pro-002"
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
