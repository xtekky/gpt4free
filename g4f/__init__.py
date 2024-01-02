from __future__ import annotations

import os

from .errors   import *
from .models   import Model, ModelUtils, _all_models
from .Provider import AsyncGeneratorProvider, ProviderUtils
from .typing   import Messages, CreateResult, AsyncResult, Union
from .         import debug, version
from .base_provider import BaseRetryProvider, ProviderType

def get_model_and_provider(model    : Union[Model, str], 
                           provider : Union[ProviderType, str, None], 
                           stream   : bool,
                           ignored  : list[str] = None,
                           ignore_working: bool = False,
                           ignore_stream: bool = False) -> tuple[str, ProviderType]:
    if debug.version_check:
        debug.version_check = False
        version.utils.check_pypi_version()
       
    if isinstance(provider, str):
        if provider in ProviderUtils.convert:
            provider = ProviderUtils.convert[provider]
        else:
            raise ProviderNotFoundError(f'Provider not found: {provider}')

    if not provider:
        if isinstance(model, str):
            if model in ModelUtils.convert:
                model = ModelUtils.convert[model]
            else:
                raise ModelNotFoundError(f'Model not found: {model}')
        provider = model.best_provider

    if not provider:
        raise ProviderNotFoundError(f'No provider found for model: {model}')
    
    if isinstance(model, Model):
        model = model.name

    if ignored and isinstance(provider, BaseRetryProvider):
        provider.providers = [p for p in provider.providers if p.__name__ not in ignored]

    if not ignore_working and not provider.working:
        raise ProviderNotWorkingError(f'{provider.__name__} is not working')
    
    if not ignore_stream and not provider.supports_stream and stream:
        raise StreamNotSupportedError(f'{provider.__name__} does not support "stream" argument')
    
    if debug.logging:
        if model:
            print(f'Using {provider.__name__} provider and {model} model')
        else:
            print(f'Using {provider.__name__} provider')

    debug.last_provider = provider

    return model, provider

class ChatCompletion:
    @staticmethod
    def create(model    : Union[Model, str],
               messages : Messages,
               provider : Union[ProviderType, str, None] = None,
               stream   : bool = False,
               auth     : Union[str, None] = None,
               ignored  : list[str] = None, 
               ignore_working: bool = False,
               ignore_stream_and_auth: bool = False,
               **kwargs) -> Union[CreateResult, str]:

        model, provider = get_model_and_provider(model, provider, stream, ignored, ignore_working, ignore_stream_and_auth)

        if not ignore_stream_and_auth and provider.needs_auth and not auth:
            raise AuthenticationRequiredError(f'{provider.__name__} requires authentication (use auth=\'cookie or token or jwt ...\' param)')

        if auth:
            kwargs['auth'] = auth
        
        if "proxy" not in kwargs:
            proxy = os.environ.get("G4F_PROXY")
            if proxy:
                kwargs['proxy'] = proxy

        result = provider.create_completion(model, messages, stream, **kwargs)
        return result if stream else ''.join(result)

    @staticmethod
    def create_async(model    : Union[Model, str],
                     messages : Messages,
                     provider : Union[ProviderType, str, None] = None,
                     stream   : bool = False,
                     ignored  : list[str] = None,
                     **kwargs) -> Union[AsyncResult, str]:

        model, provider = get_model_and_provider(model, provider, False, ignored)

        if stream:
            if isinstance(provider, type) and issubclass(provider, AsyncGeneratorProvider):
                return provider.create_async_generator(model, messages, **kwargs)
            raise StreamNotSupportedError(f'{provider.__name__} does not support "stream" argument in "create_async"')

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

        result = provider.create_completion(model, [{"role": "user", "content": prompt}], stream, **kwargs)

        return result if stream else ''.join(result)
    
def get_last_provider(as_dict: bool = False) -> Union[ProviderType, dict[str, str]]:
    last = debug.last_provider
    if isinstance(last, BaseRetryProvider):
        last = last.last_provider
    if last and as_dict:
        return {"name": last.__name__, "url": last.url}
    return last