from __future__ import annotations

import sys
from asyncio            import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from abc                import ABC, abstractmethod
from inspect            import signature, Parameter
from .helper            import get_event_loop, get_cookies, format_prompt
from ..typing           import CreateResult, AsyncResult, Messages

if sys.version_info < (3, 10):
    NoneType = type(None)
else:
    from types import NoneType

class BaseProvider(ABC):
    url: str
    working: bool = False
    needs_auth: bool = False
    supports_stream: bool = False
    supports_gpt_35_turbo: bool = False
    supports_gpt_4: bool = False
    supports_message_history: bool = False

    @staticmethod
    @abstractmethod
    def create_completion(
        model: str,
        messages: Messages,
        stream: bool,
        **kwargs
    ) -> CreateResult:
        raise NotImplementedError()

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
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
        if issubclass(cls, AsyncGeneratorProvider):
            sig = signature(cls.create_async_generator)
        elif issubclass(cls, AsyncProvider):
            sig = signature(cls.create_async)
        else:
            sig = signature(cls.create_completion)

        def get_type_name(annotation: type) -> str:
            if hasattr(annotation, "__name__"):
                annotation = annotation.__name__
            elif isinstance(annotation, NoneType):
                annotation = "None"
            return str(annotation)
        
        args = ""
        for name, param in sig.parameters.items():
            if name in ("self", "kwargs"):
                continue
            if name == "stream" and not cls.supports_stream:
                continue
            if args:
                args += ", "
            args += "\n"
            args += "    " + name
            if name != "model" and param.annotation is not Parameter.empty:
                args += f": {get_type_name(param.annotation)}"
            if param.default == "":
                args += ' = ""'
            elif param.default is not Parameter.empty:
                args += f" = {param.default}"
        
        return f"g4f.Provider.{cls.__name__} supports: ({args}\n)"


class AsyncProvider(BaseProvider):
    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
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
        messages: Messages,
        **kwargs
    ) -> str:
        raise NotImplementedError()


class AsyncGeneratorProvider(AsyncProvider):
    supports_stream = True

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
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
        messages: Messages,
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
        messages: Messages,
        **kwargs
    ) -> AsyncResult:
        raise NotImplementedError()
