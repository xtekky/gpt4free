from __future__ import annotations

from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from .helper import get_random_string, get_connector
from ..requests import raise_for_status, get_args_from_browser, WebDriver
from ..webdriver import has_seleniumwire
from ..errors import MissingRequirementsError

class GptTalkRu(AsyncGeneratorProvider):
    url = "https://gpttalk.ru"
    working = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        connector: BaseConnector = None,
        webdriver: WebDriver = None,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = "gpt-3.5-turbo"
        if not has_seleniumwire:
            raise MissingRequirementsError('Install "selenium-wire" package')
        args = get_args_from_browser(f"{cls.url}", webdriver)
        args["headers"]["accept"] = "application/json, text/plain, */*"
        async with ClientSession(connector=get_connector(connector, proxy), **args) as session:
            async with session.get("https://gpttalk.ru/getToken") as response:
                await raise_for_status(response)
                public_key = (await response.json())["response"]["key"]["publicKey"]
            random_string = get_random_string(8)
            data = {
                "model": model,
                "modelType": 1,
                "prompt": messages,
                "responseType": "stream",
                "security": {
                    "randomMessage": random_string,
                    "shifrText": encrypt(public_key, random_string)
                }
            }
            async with session.post(f"{cls.url}/gpt2", json=data, proxy=proxy) as response:
                await raise_for_status(response)
                async for chunk in response.content.iter_any():
                   yield chunk.decode()

def encrypt(public_key: str, value: str) -> str:
    from Crypto.Cipher import PKCS1_v1_5
    from Crypto.PublicKey import RSA
    import base64
    rsa_key = RSA.importKey(public_key)
    cipher = PKCS1_v1_5.new(rsa_key)
    return base64.b64encode(cipher.encrypt(value.encode())).decode()