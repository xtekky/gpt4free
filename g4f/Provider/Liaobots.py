from __future__ import annotations

import uuid
import json
from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_connector
from ..requests import raise_for_status

models = {
    "claude-3-5-sonnet-20241022": {
        "id": "claude-3-5-sonnet-20241022",
        "name": "Claude-3.5-Sonnet-V2",
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
    "claude-3-7-sonnet-20250219": {
        "id": "claude-3-7-sonnet-20250219",
        "name": "Claude-3.7-Sonnet",
        "model": "Claude",
        "provider": "Anthropic",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "claude-3-7-sonnet-20250219-t": {
        "id": "claude-3-7-sonnet-20250219-t",
        "name": "Claude-3.7-Sonnet-T",
        "model": "Claude",
        "provider": "Anthropic",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "claude-3-7-sonnet-20250219-thinking": {
        "id": "claude-3-7-sonnet-20250219-thinking",
        "name": "Claude-3.7-Sonnet-Thinking",
        "model": "Claude",
        "provider": "Anthropic",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
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
    "claude-3-sonnet-20240229": {
        "id": "claude-3-sonnet-20240229",
        "name": "Claude-3-Sonnet",
        "model": "Claude",
        "provider": "Anthropic",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "deepseek-r1": {
        "id": "deepseek-r1",
        "name": "DeepSeek-R1",
        "model": "DeepSeek-R1",
        "provider": "DeepSeek",
        "maxLength": 400000,
        "tokenLimit": 100000,
        "context": "128K",
    },
    "deepseek-r1-distill-llama-70b": {
        "id": "deepseek-r1-distill-llama-70b",
        "name": "DeepSeek-R1-70B",
        "model": "DeepSeek-R1-70B",
        "provider": "DeepSeek",
        "maxLength": 400000,
        "tokenLimit": 100000,
        "context": "128K",
    },
    "deepseek-v3": {
        "id": "deepseek-v3",
        "name": "DeepSeek-V3",
        "model": "DeepSeek-V3",
        "provider": "DeepSeek",
        "maxLength": 400000,
        "tokenLimit": 100000,
        "context": "128K",
    },
    "gemini-2.0-flash": {
        "id": "gemini-2.0-flash",
        "name": "Gemini-2.0-Flash",
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
    "gemini-2.0-pro-exp": {
        "id": "gemini-2.0-pro-exp",
        "name": "Gemini-2.0-Pro-Exp",
        "model": "Gemini",
        "provider": "Google",
        "maxLength": 4000000,
        "tokenLimit": 1000000,
        "context": "1024K",
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
    "gpt-4o-mini-2024-07-18": {
        "id": "gpt-4o-mini-2024-07-18",
        "name": "GPT-4o-Mini",
        "model": "ChatGPT",
        "provider": "OpenAI",
        "maxLength": 260000,
        "tokenLimit": 126000,
        "context": "128K",
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
    "grok-3": {
        "id": "grok-3",
        "name": "Grok-3",
        "model": "Grok",
        "provider": "x.ai",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "grok-3-r1": {
        "id": "grok-3-r1",
        "name": "Grok-3-Thinking",
        "model": "Grok",
        "provider": "x.ai",
        "maxLength": 800000,
        "tokenLimit": 200000,
        "context": "200K",
    },
    "o3-mini": {
        "id": "o3-mini",
        "name": "o3-mini",
        "model": "o3",
        "provider": "OpenAI",
        "maxLength": 400000,
        "tokenLimit": 100000,
        "context": "128K",
    },
}


class Liaobots(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://liaobots.site"
    working = True
    supports_message_history = True
    supports_system_message = True
    
    default_model = "gpt-4o-2024-08-06"
    models = list(models.keys())
    model_aliases = {
        # Anthropic
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022-t",
        "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
        "claude-3.7-sonnet": "claude-3-7-sonnet-20250219-t",
        "claude-3.7-sonnet-thinking": "claude-3-7-sonnet-20250219-thinking",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        
        # DeepSeek
        "deepseek-r1": "deepseek-r1-distill-llama-70b",
        
        # Google
        "gemini-2.0-flash-thinking": "gemini-2.0-flash-thinking-exp",
        "gemini-2.0-pro": "gemini-2.0-pro-exp",
        
        # OpenAI
        "gpt-4": default_model,
        "gpt-4o": default_model,
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-4o-mini": "gpt-4o-mini-free",       
    }
    
    _auth_code = ""
    _cookie_jar = None

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
            "referer": "https://liaobots.work/",
            "origin": "https://liaobots.work",
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
                    async for line in response.content:
                        if line.startswith(b"data: "):
                            yield json.loads(line[6:]).get("content")
            except:
                async with session.post(
                    "https://liaobots.work/api/user",
                    json={"authcode": "jGDRFOqHcZKAo"},
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
                    async for line in response.content:
                        if line.startswith(b"data: "):
                            yield json.loads(line[6:]).get("content")

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
