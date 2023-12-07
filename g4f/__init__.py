from __future__ import annotations

import os
from requests import get
from importlib.metadata import version as get_package_version, PackageNotFoundError
from subprocess import check_output, CalledProcessError, PIPE

from .models   import Model, ModelUtils, _all_models
from .Provider import BaseProvider, AsyncGeneratorProvider, RetryProvider
from .typing   import Messages, CreateResult, AsyncResult, Union, List
from .         import debug

def get_version() -> str:
    # Read from package manager
    try:
        return get_package_version("g4f")
    except PackageNotFoundError:
        pass
    # Read from docker environment
    current_version = os.environ.get("G4F_VERSION")
    if current_version:
        return current_version
    # Read from git repository
    try:
        command = ["git", "describe", "--tags", "--abbrev=0"]
        return check_output(command, text=True, stderr=PIPE).strip()
    except CalledProcessError:
        pass
    
def get_lastet_version() -> str:
    response = get("https://pypi.org/pypi/g4f/json").json()
    return response["info"]["version"]

def check_pypi_version() -> None:
    try:
        version = get_version()
        latest_version = get_lastet_version()
    except Exception as e:
        print(f'Failed to check g4f pypi version: {e}')
    if version != latest_version:
        print(f'New pypi version: {latest_version} (current: {version}) | pip install -U g4f')

def get_model_and_provider(model    : Union[Model, str], 
                           provider : Union[type[BaseProvider], None], 
                           stream   : bool,
                           ignored  : List[str] = None,
                           ignore_working: bool = False,
                           ignore_stream: bool = False) -> tuple[Model, type[BaseProvider]]:
    if debug.version_check:
        check_pypi_version()
        debug.version_check = False
    
    if isinstance(model, str):
        if model in ModelUtils.convert:
            model = ModelUtils.convert[model]
        else:
            raise ValueError(f'The model: {model} does not exist')

    if not provider:
        provider = model.best_provider

    if isinstance(provider, RetryProvider) and ignored:
        provider.providers = [p for p in provider.providers if p.__name__ not in ignored]

    if not provider:
        raise RuntimeError(f'No provider found for model: {model}')

    if not provider.working and not ignore_working:
        raise RuntimeError(f'{provider.__name__} is not working')

    if not ignore_stream and not provider.supports_stream and stream:
        raise ValueError(f'{provider.__name__} does not support "stream" argument')

    if debug.logging:
        print(f'Using {provider.__name__} provider')

    return model, provider

class ChatCompletion:
    @staticmethod
    def create(model    : Union[Model, str],
               messages : Messages,
               provider : Union[type[BaseProvider], None] = None,
               stream   : bool = False,
               auth     : Union[str, None] = None,
               ignored  : List[str] = None, 
               ignore_working: bool = False,
               ignore_stream_and_auth: bool = False,
               **kwargs) -> Union[CreateResult, str]:

        model, provider = get_model_and_provider(model, provider, stream, ignored, ignore_working, ignore_stream_and_auth)

        if not ignore_stream_and_auth and provider.needs_auth and not auth:
            raise ValueError(
                f'{provider.__name__} requires authentication (use auth=\'cookie or token or jwt ...\' param)')

        if auth:
            kwargs['auth'] = auth

        result = provider.create_completion(model.name, messages, stream, **kwargs)
        return result if stream else ''.join(result)

    @staticmethod
    async def create_async(model    : Union[Model, str],
                           messages : Messages,
                           provider : Union[type[BaseProvider], None] = None,
                           stream   : bool = False,
                           ignored  : List[str] = None,
                           **kwargs) -> Union[AsyncResult, str]:
        model, provider = get_model_and_provider(model, provider, False, ignored)

        if stream:
            if isinstance(provider, type) and issubclass(provider, AsyncGeneratorProvider):
                return await provider.create_async_generator(model.name, messages, **kwargs)
            raise ValueError(f'{provider.__name__} does not support "stream" argument')

        return await provider.create_async(model.name, messages, **kwargs)

class Completion:
    @staticmethod
    def create(model    : Union[Model, str],
               prompt   : str,
               provider : Union[type[BaseProvider], None] = None,
               stream   : bool = False,
               ignored  : List[str] = None, **kwargs) -> Union[CreateResult, str]:

        allowed_models = [
            'code-davinci-002',
            'text-ada-001',
            'text-babbage-001',
            'text-curie-001',
            'text-davinci-002',
            'text-davinci-003'
        ]

        if model not in allowed_models:
            raise Exception(f'ValueError: Can\'t use {model} with Completion.create()')

        model, provider = get_model_and_provider(model, provider, stream, ignored)

        result = provider.create_completion(model.name, [{"role": "user", "content": prompt}], stream, **kwargs)

        return result if stream else ''.join(result)