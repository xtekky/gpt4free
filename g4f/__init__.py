from __future__ import annotations
from requests import get
from g4f.models import Model, ModelUtils
from .Provider import BaseProvider
from .typing import Messages, CreateResult, Union
from .debug import logging

version = '0.1.5.7'
version_check = True

def check_pypi_version() -> None:
    try:
        response = get("https://pypi.org/pypi/g4f/json").json()
        latest_version = response["info"]["version"]

        if version != latest_version:
            print(f'New pypi version: {latest_version} (current: {version}) | pip install -U g4f')

    except Exception as e:
        print(f'Failed to check g4f pypi version: {e}')

def get_model_and_provider(model    : Union[Model, str], 
                           provider : Union[type[BaseProvider], None], 
                           stream   : bool) -> tuple[Model, type[BaseProvider]]:
    
    if isinstance(model, str):
        if model in ModelUtils.convert:
            model = ModelUtils.convert[model]
        else:
            raise ValueError(f'The model: {model} does not exist')

    if not provider:
        provider = model.best_provider

    if not provider:
        raise RuntimeError(f'No provider found for model: {model}')

    if not provider.working:
        raise RuntimeError(f'{provider.__name__} is not working')

    if not provider.supports_stream and stream:
        raise ValueError(f'{provider.__name__} does not support "stream" argument')

    if logging:
        print(f'Using {provider.__name__} provider')

    return model, provider

class ChatCompletion:
    @staticmethod
    def create(
        model: Union[Model, str],
        messages : Messages,
        provider : Union[type[BaseProvider], None] = None,
        stream   : bool = False,
        auth     : Union[str, None] = None,
        **kwargs
    ) -> Union[CreateResult, str]:

        model, provider = get_model_and_provider(model, provider, stream)

        if provider.needs_auth and not auth:
            raise ValueError(
                f'{provider.__name__} requires authentication (use auth=\'cookie or token or jwt ...\' param)')

        if provider.needs_auth:
            kwargs['auth'] = auth

        result = provider.create_completion(model.name, messages, stream, **kwargs)
        return result if stream else ''.join(result)

    @staticmethod
    async def create_async(
        model: Union[Model, str],
        messages: Messages,
        provider: Union[type[BaseProvider], None] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        if stream:
            raise ValueError(f'"create_async" does not support "stream" argument')

        model, provider = get_model_and_provider(model, provider, False)

        return await provider.create_async(model.name, messages, **kwargs)

class Completion:
    @staticmethod
    def create(
        model: str,
        prompt: str,
        provider: Union[type[BaseProvider], None] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[CreateResult, str]:

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

        model, provider = get_model_and_provider(model, provider, stream)

        result = provider.create_completion(model.name, [{"role": "user", "content": prompt}], stream, **kwargs)

        return result if stream else ''.join(result)
    
if version_check:
    check_pypi_version()