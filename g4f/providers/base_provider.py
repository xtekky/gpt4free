from __future__ import annotations

import sys
import asyncio
from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from abc import abstractmethod
from inspect import signature, Parameter
from typing import Callable, Union
from ..typing import CreateResult, AsyncResult, Messages
from .types import BaseProvider, FinishReason
from ..errors import NestAsyncioError, ModelNotSupportedError
from .. import debug

if sys.version_info < (3, 10):
    NoneType = type(None)
else:
    from types import NoneType

# Set Windows event loop policy for better compatibility with asyncio and curl_cffi
if sys.platform == 'win32':
    if isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def get_running_loop(check_nested: bool) -> Union[AbstractEventLoop, None]:
    try:
        loop = asyncio.get_running_loop()
        if check_nested and not hasattr(loop.__class__, "_nest_patched"):
            try:
                import nest_asyncio
                nest_asyncio.apply(loop)
            except ImportError:
                raise NestAsyncioError('Install "nest_asyncio" package')
        return loop
    except RuntimeError:
        pass

# Fix for RuntimeError: async generator ignored GeneratorExit
async def await_callback(callback: Callable):
    return await callback()

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
        loop = loop or asyncio.get_running_loop()

        def create_func() -> str:
            return "".join(cls.create_completion(model, messages, False, **kwargs))

        return await asyncio.wait_for(
            loop.run_in_executor(executor, create_func),
            timeout=kwargs.get("timeout")
        )

    def get_parameters(cls) -> dict:
        return signature(
            cls.create_async_generator if issubclass(cls, AsyncGeneratorProvider) else
            cls.create_async if issubclass(cls, AsyncProvider) else
            cls.create_completion
        ).parameters

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

        def get_type_name(annotation: type) -> str:
            return annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)

        args = ""
        for name, param in cls.get_parameters().items():
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
        get_running_loop(check_nested=True)
        yield asyncio.run(cls.create_async(model, messages, **kwargs))

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
        loop = get_running_loop(check_nested=True)
        new_loop = False
        if loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            new_loop = True

        generator = cls.create_async_generator(model, messages, stream=stream, **kwargs)
        gen = generator.__aiter__()

        try:
            while True:
                yield loop.run_until_complete(await_callback(gen.__anext__))
        except StopAsyncIteration:
            ...
        finally:
            if new_loop:
                loop.close()
                asyncio.set_event_loop(None)

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
            if not isinstance(chunk, (Exception, FinishReason))
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
    
class ProviderModelMixin:
    default_model: str
    models: list[str] = []
    model_aliases: dict[str, str] = {}
    
    @classmethod
    def get_models(cls) -> list[str]:
        return cls.models
    
    @classmethod
    def get_model(cls, model: str) -> str:
        if not model and cls.default_model is not None:
            model = cls.default_model
        elif model in cls.model_aliases:
            model = cls.model_aliases[model]
        elif model not in cls.get_models() and cls.models:
            raise ModelNotSupportedError(f"Model is not supported: {model} in: {cls.__name__}")
        debug.last_model = model
        return model