from __future__ import annotations
from requests   import get
from .models    import Model, ModelUtils, _all_models
from .Provider  import BaseProvider, AsyncGeneratorProvider, RetryProvider
from .typing    import Messages, CreateResult, AsyncResult, Union, List
from .          import debug

version       = '0.1.9.1'
version_check = True

def check_pypi_version() -> None:
    try:
        response = get("https://pypi.org/pypi/g4f/json").json()
        latest_version = response["info"]["version"]

        if version != latest_version:
            print(f'New pypi version: {latest_version} (current: {version}) | pip install -U g4f')
            return False
        return True

    except Exception as e:
        print(f'Failed to check g4f pypi version: {e}')

def get_model_and_provider(model    : Union[Model, str], 
                           provider : Union[type[BaseProvider], None], 
                           stream   : bool,
                           ignored  : List[str] = None,
                           ignore_working: bool = False) -> tuple[Model, type[BaseProvider]]:
    
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

    if not provider.supports_stream and stream:
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
               ignore_working: bool = False, **kwargs) -> Union[CreateResult, str]:

        model, provider = get_model_and_provider(model, provider, stream, ignored, ignore_working)

        if provider.needs_auth and not auth:
            raise ValueError(
                f'{provider.__name__} requires authentication (use auth=\'cookie or token or jwt ...\' param)')

        if provider.needs_auth:
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
    
if version_check:
    check_pypi_version()