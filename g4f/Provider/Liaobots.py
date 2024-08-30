from __future__ import annotations

import uuid
import requests
from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_connector
from ..requests import raise_for_status

class Liaobots(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://liaobots.site"
    working = True
    supports_message_history = True
    supports_system_message = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    default_model = "gpt-4o"
    models = None
    model_aliases = {
        "gpt-4o-mini": "gpt-4o-mini-free",
        "gpt-4o": "gpt-4o-free",
        "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
        "gpt-4o": "gpt-4o-2024-08-06",
        "gpt-4": "gpt-4-0613",

        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-opus": "claude-3-opus-20240229-aws",
        "claude-3-opus": "claude-3-opus-20240229-gcp",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-2.1": "claude-2.1",

        "gemini-pro": "gemini-1.0-pro-latest",
        "gemini-flash": "gemini-1.5-flash-latest",
        "gemini-pro": "gemini-1.5-pro-latest",
    }
    _auth_code = ""
    _cookie_jar = None

    @classmethod
    def get_models(cls):
        if cls.models is None:
            url = 'https://liaobots.work/api/models'
            headers = {
                'accept': '/',
                'accept-language': 'en-US,en;q=0.9',
                'content-type': 'application/json',
                'cookie': 'gkp2=ehnhUPJtkCgMmod8Sbxn',
                'origin': 'https://liaobots.work',
                'priority': 'u=1, i',
                'referer': 'https://liaobots.work/',
                'sec-ch-ua': '"Chromium";v="127", "Not)A;Brand";v="99"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Linux"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'
            }
            data = {'key': ''}

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                try:
                    models_data = response.json()
                    cls.models = {model['id']: model for model in models_data}
                except (ValueError, KeyError) as e:
                    print(f"Error processing JSON response: {e}")
                    cls.models = {}
            else:
                print(f"Request failed with status code: {response.status_code}")
                cls.models = {}

        return cls.models

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
            models = cls.get_models()
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
                await cls.ensure_auth_code(session)
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
                await cls.initialize_auth_code(session)
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
        models = cls.get_models()
        if model not in models:
            raise ValueError(f"Model '{model}' is not supported.")
        return model
    @classmethod
    def is_supported(cls, model: str) -> bool:
        """
        Check if the given model is supported.
        """
        models = cls.get_models()
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

    @classmethod
    async def refresh_auth_code(cls, session: ClientSession) -> None:
        """
        Refresh the auth code by making a new request.
        """
        await cls.initialize_auth_code(session)

    @classmethod
    async def get_auth_code(cls, session: ClientSession) -> str:
        """
        Get the current auth code, initializing it if necessary.
        """
        await cls.ensure_auth_code(session)
        return cls._auth_code