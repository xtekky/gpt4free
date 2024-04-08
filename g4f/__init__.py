from __future__ import annotations

import os

from . import debug, version
from .models import Model
from .typing import Messages, CreateResult, AsyncResult, Union
from .errors import StreamNotSupportedError, ModelNotAllowedError
from .cookies import get_cookies, set_cookies
from .providers.types import ProviderType
from .providers.base_provider import AsyncGeneratorProvider
from .client.service import get_model_and_provider, get_last_provider

class ChatCompletion:
    @staticmethod
    def create(model    : Union[Model, str],
               messages : Messages,
               provider : Union[ProviderType, str, None] = None,
               stream   : bool = False,
               auth     : Union[str, None] = None,
               ignored  : list[str] = None, 
               ignore_working: bool = False,
               ignore_stream: bool = False,
               patch_provider: callable = None,
               **kwargs) -> Union[CreateResult, str]:
        """
        Creates a chat completion using the specified model, provider, and messages.

        Args:
            model (Union[Model, str]): The model to use, either as an object or a string identifier.
            messages (Messages): The messages for which the completion is to be created.
            provider (Union[ProviderType, str, None], optional): The provider to use, either as an object, a string identifier, or None.
            stream (bool, optional): Indicates if the operation should be performed as a stream.
            auth (Union[str, None], optional): Authentication token or credentials, if required.
            ignored (list[str], optional): List of provider names to be ignored.
            ignore_working (bool, optional): If True, ignores the working status of the provider.
            ignore_stream (bool, optional): If True, ignores the stream and authentication requirement checks.
            patch_provider (callable, optional): Function to modify the provider.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[CreateResult, str]: The result of the chat completion operation.

        Raises:
            AuthenticationRequiredError: If authentication is required but not provided.
            ProviderNotFoundError, ModelNotFoundError: If the specified provider or model is not found.
            ProviderNotWorkingError: If the provider is not operational.
            StreamNotSupportedError: If streaming is requested but not supported by the provider.
        """
        model, provider = get_model_and_provider(
            model, provider, stream,
            ignored, ignore_working,
            ignore_stream or kwargs.get("ignore_stream_and_auth")
        )

        if auth is not None:
            kwargs['auth'] = auth
        
        if "proxy" not in kwargs:
            proxy = os.environ.get("G4F_PROXY")
            if proxy:
                kwargs['proxy'] = proxy

        if patch_provider:
            provider = patch_provider(provider)

        result = provider.create_completion(model, messages, stream, **kwargs)
        return result if stream else ''.join([str(chunk) for chunk in result])

    @staticmethod
    def create_async(model    : Union[Model, str],
                     messages : Messages,
                     provider : Union[ProviderType, str, None] = None,
                     stream   : bool = False,
                     ignored  : list[str] = None,
                     ignore_working: bool = False,
                     patch_provider: callable = None,
                     **kwargs) -> Union[AsyncResult, str]:
        """
        Asynchronously creates a completion using the specified model and provider.

        Args:
            model (Union[Model, str]): The model to use, either as an object or a string identifier.
            messages (Messages): Messages to be processed.
            provider (Union[ProviderType, str, None]): The provider to use, either as an object, a string identifier, or None.
            stream (bool): Indicates if the operation should be performed as a stream.
            ignored (list[str], optional): List of provider names to be ignored.
            patch_provider (callable, optional): Function to modify the provider.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[AsyncResult, str]: The result of the asynchronous chat completion operation.

        Raises:
            StreamNotSupportedError: If streaming is requested but not supported by the provider.
        """
        model, provider = get_model_and_provider(model, provider, False, ignored, ignore_working)

        if stream:
            if isinstance(provider, type) and issubclass(provider, AsyncGeneratorProvider):
                return provider.create_async_generator(model, messages, **kwargs)
            raise StreamNotSupportedError(f'{provider.__name__} does not support "stream" argument in "create_async"')

        if patch_provider:
            provider = patch_provider(provider)

        return provider.create_async(model, messages, **kwargs)

class Completion:
    @staticmethod
    def create(model    : Union[Model, str],
               prompt   : str,
               provider : Union[ProviderType, None] = None,
               stream   : bool = False,
               ignored  : list[str] = None, **kwargs) -> Union[CreateResult, str]:
        """
        Creates a completion based on the provided model, prompt, and provider.

        Args:
            model (Union[Model, str]): The model to use, either as an object or a string identifier.
            prompt (str): The prompt text for which the completion is to be created.
            provider (Union[ProviderType, None], optional): The provider to use, either as an object or None.
            stream (bool, optional): Indicates if the operation should be performed as a stream.
            ignored (list[str], optional): List of provider names to be ignored.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[CreateResult, str]: The result of the completion operation.

        Raises:
            ModelNotAllowedError: If the specified model is not allowed for use with this method.
        """
        allowed_models = [
            'code-davinci-002',
            'text-ada-001',
            'text-babbage-001',
            'text-curie-001',
            'text-davinci-002',
            'text-davinci-003'
        ]
        if model not in allowed_models:
            raise ModelNotAllowedError(f'Can\'t use {model} with Completion.create()')

        model, provider = get_model_and_provider(model, provider, stream, ignored)

        result = provider.create_completion(model, [{"role": "user", "content": prompt}], stream, **kwargs)

        return result if stream else ''.join(result)