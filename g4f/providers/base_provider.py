from __future__ import annotations

import asyncio

from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from abc import abstractmethod
from inspect import signature, Parameter
from typing import Optional, _GenericAlias
try:
    from types import NoneType
except ImportError:
    NoneType = type(None)

from ..typing import CreateResult, AsyncResult, Messages
from .types import BaseProvider
from .asyncio import get_running_loop, to_sync_generator
from .response import BaseConversation
from .helper import concat_chunks, async_concat_chunks
from ..errors import ModelNotSupportedError
from .. import debug

SAFE_PARAMETERS = [
    "model", "messages", "stream", "timeout",
    "proxy", "images", "response_format",
    "prompt", "tools", "conversation",
    "history_disabled", "auto_continue",
    "temperature",  "top_k", "top_p",
    "frequency_penalty", "presence_penalty",
    "max_tokens", "max_new_tokens", "stop",
    "api_key", "seed", "width", "height",
    "proof_token", "max_retries"
]

BASIC_PARAMETERS = {
    "model": "",
    "messages": [],
    "provider": None,
    "stream": False,
    "timeout": 0,
    "response_format": None,
    "max_tokens": None,
    "stop": None,
    "web_search": False,
}

PARAMETER_EXAMPLES = {
    "proxy": "http://user:password@127.0.0.1:3128",
    "temperature": 1,
    "top_k": 1,
    "top_p": 1,
    "frequency_penalty": 1,
    "presence_penalty": 1,
    "messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
    "images": [["data:image/jpeg;base64,...", "filename.jpg"]],
    "response_format": {"type": "json_object"},
    "conversation": {"conversation_id": "550e8400-e29b-11d4-a716-...", "message_id": "550e8400-e29b-11d4-a716-..."},
    "max_new_tokens": 1024,
    "max_tokens": 4096,
    "seed": 42,
}

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
        timeout: int = None,
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
            return concat_chunks(cls.create_completion(model, messages, False, **kwargs))

        return await asyncio.wait_for(
            loop.run_in_executor(executor, create_func),
            timeout=timeout
        )

    @classmethod
    def get_parameters(cls, as_json: bool = False) -> dict[str, Parameter]:
        params = {name: parameter for name, parameter in signature(
            cls.create_async_generator if issubclass(cls, AsyncGeneratorProvider) else
            cls.create_async if issubclass(cls, AsyncProvider) else
            cls.create_completion
        ).parameters.items() if name in SAFE_PARAMETERS
            and (name != "stream" or cls.supports_stream)}
        if as_json:
            def get_type_as_var(annotation: type, key: str):
                if key == "model":
                    return getattr(cls, "default_model", "")
                elif key == "stream":
                    return cls.supports_stream
                elif key in PARAMETER_EXAMPLES:
                    if key == "messages" and not cls.supports_system_message:
                        return [PARAMETER_EXAMPLES[key][-1]]
                    return PARAMETER_EXAMPLES[key]
                if isinstance(annotation, type):
                    if issubclass(annotation, int):
                        return 0
                    elif issubclass(annotation, float):
                        return 0.0
                    elif issubclass(annotation, bool):
                        return False
                    elif issubclass(annotation, str):
                        return ""
                    elif issubclass(annotation, dict):
                        return {}
                    elif issubclass(annotation, list):
                        return []
                    elif issubclass(annotation, BaseConversation):
                        return {}
                    elif issubclass(annotation, NoneType):
                        return {}
                elif annotation is None:
                    return None
                elif isinstance(annotation, _GenericAlias) and annotation.__origin__ is Optional:
                    return get_type_as_var(annotation.__args__[0])
                else:
                    return str(annotation)
            return { name: (
                param.default
                if isinstance(param, Parameter) and param.default is not Parameter.empty and param.default is not None
                else get_type_as_var(param.annotation if isinstance(param, Parameter) else type(param), name)
            ) for name, param in {
                **BASIC_PARAMETERS,
                **{"provider": cls.__name__},
                **params
            }.items()}
        return params

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
            return getattr(annotation, "__name__", str(annotation)) if annotation is not Parameter.empty else ""

        args = ""
        for name, param in cls.get_parameters().items():
            args += f"\n    {name}"
            args += f": {get_type_name(param.annotation)}"
            default_value = getattr(cls, "default_model", "") if name == "model" else param.default
            default_value = f'"{default_value}"' if isinstance(default_value, str) else default_value
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
        return await async_concat_chunks(cls.create_async_generator(model, messages, stream=False, **kwargs))

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