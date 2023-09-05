from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import browser_cookie3

from ..typing import Any, AsyncGenerator, CreateResult, Union


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
        try:
            for cookie in browser_cookie3.load(cookie_domain):
                _cookies[cookie_domain][cookie.name] = cookie.value
        except:
            pass
    return _cookies[cookie_domain]


def format_prompt(messages: list[dict[str, str]], add_special_tokens=False):
    if add_special_tokens or len(messages) > 1:
        formatted = "\n".join(
            ["%s: %s" % ((message["role"]).capitalize(), message["content"]) for message in messages]
        )
        return f"{formatted}\nAssistant:"
    else:
        return messages.pop()["content"]



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
    supports_stream = True

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = True,
        **kwargs
    ) -> CreateResult:
        yield from run_generator(cls.create_async_generator(model, messages, stream=stream, **kwargs))

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: list[dict[str, str]],
        **kwargs
    ) -> str:
        chunks = [chunk async for chunk in cls.create_async_generator(model, messages, stream=False, **kwargs)]
        if chunks:
            return "".join(chunks)
        
    @staticmethod
    @abstractmethod
    def create_async_generator(
            model: str,
            messages: list[dict[str, str]],
            **kwargs
        ) -> AsyncGenerator:
        raise NotImplementedError()


def run_generator(generator: AsyncGenerator[Union[Any, str], Any]):
    loop = asyncio.new_event_loop()
    gen  = generator.__aiter__()

    while True:
        try:
            yield loop.run_until_complete(gen.__anext__())

        except StopAsyncIteration:
            break
