from __future__ import annotations
import sys
import asyncio
from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from abc import abstractmethod
from inspect import signature, Parameter
from .helper import get_event_loop, get_cookies, format_prompt
from ..typing import CreateResult, AsyncResult, Messages
from ..base_provider import BaseProvider

if sys.version_info < (3, 10):
    NoneType = type(None)
else:
    from types import NoneType

# Set Windows event loop policy for better compatibility with asyncio and curl_cffi
if sys.platform == 'win32':
    if isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class AbstractProvider(BaseProvider):
    """
    Abstract class for providing asynchronous functionality to derived classes.
    """

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
        """
        Asynchronously creates a result based on the given model and messages.

        Args:
            cls (type): The class on which this method is called.
            model (str): The model to use for creation.
            messages (Messages): The messages to process.
            loop (AbstractEventLoop, optional): The event loop to use. Defaults to None.
            executor (ThreadPoolExecutor, optional): The executor for running async tasks. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The created result as a string.
        """
        loop = loop or get_event_loop()

        def create_func() -> str:
            return "".join(cls.create_completion(model, messages, False, **kwargs))

        return await asyncio.wait_for(
            loop.run_in_executor(executor, create_func),
            timeout=kwargs.get("timeout", 0)
        )
    
    @classmethod
    @property
    def params(cls) -> str:
        """
        Returns the parameters supported by the provider.

        Args:
            cls (type): The class on which this property is called.

        Returns:
            str: A string listing the supported parameters.
        """
        sig = signature(
            cls.create_async_generator if issubclass(cls, AsyncGeneratorProvider) else
            cls.create_async if issubclass(cls, AsyncProvider) else
            cls.create_completion
        )

        def get_type_name(annotation: type) -> str:
            return annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)

        args = ""
        for name, param in sig.parameters.items():
            if name in ("self", "kwargs") or (name == "stream" and not cls.supports_stream):
                continue
            args += f"\n    {name}"
            args += f": {get_type_name(param.annotation)}" if param.annotation is not Parameter.empty else ""
            args += f' = "{param.default}"' if param.default == "" else f" = {param.default}" if param.default is not Parameter.empty else ""
        
        return f"g4f.Provider.{cls.__name__} supports: ({args}\n)"


class AsyncProvider(AbstractProvider):
    """
    Provides asynchronous functionality for creating completions.
    """

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        *,
        loop: AbstractEventLoop = None,
        **kwargs
    ) -> CreateResult:
        """
        Creates a completion result synchronously.

        Args:
            cls (type): The class on which this method is called.
            model (str): The model to use for creation.
            messages (Messages): The messages to process.
            stream (bool): Indicates whether to stream the results. Defaults to False.
            loop (AbstractEventLoop, optional): The event loop to use. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            CreateResult: The result of the completion creation.
        """
        loop = loop or get_event_loop()
        coro = cls.create_async(model, messages, **kwargs)
        yield loop.run_until_complete(coro)

    @staticmethod
    @abstractmethod
    async def create_async(
        model: str,
        messages: Messages,
        **kwargs
    ) -> str:
        """
        Abstract method for creating asynchronous results.

        Args:
            model (str): The model to use for creation.
            messages (Messages): The messages to process.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: If this method is not overridden in derived classes.

        Returns:
            str: The created result as a string.
        """
        raise NotImplementedError()


class AsyncGeneratorProvider(AsyncProvider):
    """
    Provides asynchronous generator functionality for streaming results.
    """
    supports_stream = True

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        *,
        loop: AbstractEventLoop = None,
        **kwargs
    ) -> CreateResult:
        """
        Creates a streaming completion result synchronously.

        Args:
            cls (type): The class on which this method is called.
            model (str): The model to use for creation.
            messages (Messages): The messages to process.
            stream (bool): Indicates whether to stream the results. Defaults to True.
            loop (AbstractEventLoop, optional): The event loop to use. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            CreateResult: The result of the streaming completion creation.
        """
        loop = loop or get_event_loop()
        generator = cls.create_async_generator(model, messages, stream=stream, **kwargs)
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
        """
        Asynchronously creates a result from a generator.

        Args:
            cls (type): The class on which this method is called.
            model (str): The model to use for creation.
            messages (Messages): The messages to process.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The created result as a string.
        """
        return "".join([
            chunk async for chunk in cls.create_async_generator(model, messages, stream=False, **kwargs) 
            if not isinstance(chunk, Exception)
        ])

    @staticmethod
    @abstractmethod
    async def create_async_generator(
        model: str,
        messages: Messages,
        stream: bool = True,
        **kwargs
    ) -> AsyncResult:
        """
        Abstract method for creating an asynchronous generator.

        Args:
            model (str): The model to use for creation.
            messages (Messages): The messages to process.
            stream (bool): Indicates whether to stream the results. Defaults to True.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: If this method is not overridden in derived classes.

        Returns:
            AsyncResult: An asynchronous generator yielding results.
        """
        raise NotImplementedError()