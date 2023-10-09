from __future__ import annotations

from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod

from .helper import get_event_loop, get_cookies, format_prompt
from ..typing import AsyncGenerator, CreateResult


class BaseProvider(ABC):
    url: str
    working: bool = False
    needs_auth: bool = False
    supports_stream: bool = False
    supports_gpt_35_turbo: bool = False
    supports_gpt_4: bool = False

    @staticmethod
    @abstractmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs
    ) -> CreateResult:
        raise NotImplementedError()

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: list[dict[str, str]],
        *,
        loop: AbstractEventLoop = None,
        executor: ThreadPoolExecutor = None,
        **kwargs
    ) -> str:
        if not loop:
            loop = get_event_loop()

        def create_func() -> str:
            return "".join(cls.create_completion(
                model,
                messages,
                False,
                **kwargs
            ))

        return await loop.run_in_executor(
            executor,
            create_func
        )

    @classmethod
    @property
    def params(cls) -> str:
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"


class AsyncProvider(BaseProvider):
    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> CreateResult:
        loop = get_event_loop()
        coro = cls.create_async(model, messages, **kwargs)
        yield loop.run_until_complete(coro)

    @staticmethod
    @abstractmethod
    async def create_async(
        model: str,
        messages: list[dict[str, str]],
        **kwargs
    ) -> str:
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
        loop = get_event_loop()
        generator = cls.create_async_generator(
            model,
            messages,
            stream=stream,
            **kwargs
        )
        gen = generator.__aiter__()
        while True:
            try:
                yield loop.run_until_complete(gen.__anext__())
            except StopAsyncIteration:
                break

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: list[dict[str, str]],
        **kwargs
    ) -> str:
        return "".join([
            chunk async for chunk in cls.create_async_generator(
                model,
                messages,
                stream=False,
                **kwargs
            )
        ])

    @staticmethod
    @abstractmethod
    def create_async_generator(
        model: str,
        messages: list[dict[str, str]],
        **kwargs
    ) -> AsyncGenerator:
        raise NotImplementedError()