from __future__ import annotations

import os
import logging

from . import debug, version
from .models import Model
from .client import Client, AsyncClient
from .typing import Messages, CreateResult, AsyncResult, Union
from .errors import StreamNotSupportedError, ModelNotAllowedError
from .cookies import get_cookies, set_cookies
from .providers.types import ProviderType
from .providers.base_provider import AsyncGeneratorProvider
from .client.service import get_model_and_provider, get_last_provider

#Configure "g4f" logger
logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler()
log_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
logger.addHandler(log_handler)

logger.setLevel(logging.ERROR)

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

        result = provider.create_completion(model, messages, stream=stream, **kwargs)

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

        result = provider.create_completion(model, [{"role": "user", "content": prompt}], stream=stream, **kwargs)

        return result if stream else ''.join(result)
