from abc import ABC, abstractmethod

from ..typing import Any, CreateResult, AsyncGenerator, Union

import browser_cookie3
import asyncio
from time import time
import math

class BaseProvider(ABC):
    url: str
    working               = False
    needs_auth            = False
    supports_stream       = False
    supports_gpt_35_turbo = False
    supports_gpt_4        = False

    @staticmethod
    @abstractmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:
        
        raise NotImplementedError()

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
    

_cookies = {}

def get_cookies(cookie_domain: str) -> dict:
    if cookie_domain not in _cookies:
        _cookies[cookie_domain] = {}
        
        for cookie in browser_cookie3.load(cookie_domain):
            _cookies[cookie_domain][cookie.name] = cookie.value
    
    return _cookies[cookie_domain]


class AsyncProvider(BaseProvider):
    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False, **kwargs: Any) -> CreateResult:
        
        yield asyncio.run(cls.create_async(model, messages, **kwargs))

    @staticmethod
    @abstractmethod
    async def create_async(
        model: str,
        messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError()


class AsyncGeneratorProvider(AsyncProvider):
    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = True, **kwargs: Any) -> CreateResult:
        
        if stream:
            yield from run_generator(cls.create_async_generator(model, messages, **kwargs))
        else:
            yield from AsyncProvider.create_completion(cls=cls, model=model, messages=messages, **kwargs)

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: list[dict[str, str]], **kwargs: Any) -> str:
        
        chunks = [chunk async for chunk in cls.create_async_generator(model, messages, **kwargs)]
        if chunks:
            return "".join(chunks)
        
    @staticmethod
    @abstractmethod
    def create_async_generator(
            model: str,
            messages: list[dict[str, str]]) -> AsyncGenerator:
        
        raise NotImplementedError()


def run_generator(generator: AsyncGenerator[Union[Any, str], Any]):
    loop = asyncio.new_event_loop()
    gen  = generator.__aiter__()

    while True:
        try:
            yield loop.run_until_complete(gen.__anext__())

        except StopAsyncIteration:
            break
