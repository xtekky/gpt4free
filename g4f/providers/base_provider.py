from __future__ import annotations

import sys
import asyncio

from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from abc import abstractmethod
from inspect import signature, Parameter

from ..typing import CreateResult, AsyncResult, Messages
from .types import BaseProvider
from .asyncio import get_running_loop, to_sync_generator
from .response import FinishReason, BaseConversation, SynthesizeData
from ..errors import ModelNotSupportedError
from .. import debug

# Set Windows event loop policy for better compatibility with asyncio and curl_cffi
if sys.platform == 'win32':
    try:
        from curl_cffi import aio
        if not hasattr(aio, "_get_selector"):
            if isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except ImportError:
        pass

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
            chunks = [str(chunk) for chunk in cls.create_completion(model, messages, False, **kwargs) if chunk]
            if chunks:
                return "".join(chunks)

        return await asyncio.wait_for(
            loop.run_in_executor(executor, create_func),
            timeout=kwargs.get("timeout")
        )

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:
        return {name: parameter for name, parameter in signature(
            cls.create_async_generator if issubclass(cls, AsyncGeneratorProvider) else
            cls.create_async if issubclass(cls, AsyncProvider) else
            cls.create_completion
        ).parameters.items() if name not in ["kwargs", "model", "messages"]
            and (name != "stream" or cls.supports_stream)}

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
            args += f"\n    {name}"
            args += f": {get_type_name(param.annotation)}" if param.annotation is not Parameter.empty else ""
            default_value = f'"{param.default}"' if isinstance(param.default, str) else param.default
            args += f" = {default_value}" if param.default is not Parameter.empty else ""
            args += ","

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
        get_running_loop(check_nested=False)
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
        return to_sync_generator(
            cls.create_async_generator(model, messages, stream=stream, **kwargs)
        )

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
            str(chunk) async for chunk in cls.create_async_generator(model, messages, stream=False, **kwargs) 
            if chunk and not isinstance(chunk, (Exception, FinishReason, BaseConversation, SynthesizeData))
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
    default_model: str = None
    models: list[str] = []
    model_aliases: dict[str, str] = {}
    image_models: list = None
    last_model: str = None

    @classmethod
    def get_models(cls, **kwargs) -> list[str]:
        if not cls.models and cls.default_model is not None:
            return [cls.default_model]
        return cls.models

    @classmethod
    def get_model(cls, model: str, **kwargs) -> str:
        if not model and cls.default_model is not None:
            model = cls.default_model
        elif model in cls.model_aliases:
            model = cls.model_aliases[model]
        else:
            if model not in cls.get_models(**kwargs) and cls.models:
                raise ModelNotSupportedError(f"Model is not supported: {model} in: {cls.__name__}")
        cls.last_model = model
        debug.last_model = model
        return model