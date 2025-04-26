from __future__ import annotations

from typing import Union

from .. import debug, version
from ..errors import ProviderNotFoundError, ModelNotFoundError, ProviderNotWorkingError, StreamNotSupportedError
from ..models import Model, ModelUtils, default, default_vision
from ..Provider import ProviderUtils
from ..providers.types import BaseRetryProvider, ProviderType
from ..providers.retry_provider import IterListProvider

def convert_to_provider(provider: str) -> ProviderType:
    if " " in provider:
        provider_list = [ProviderUtils.convert[p] for p in provider.split() if p in ProviderUtils.convert]
        if not provider_list:
            raise ProviderNotFoundError(f'Providers not found: {provider}')
        provider = IterListProvider(provider_list, False)
    elif provider in ProviderUtils.convert:
        provider = ProviderUtils.convert[provider]
    elif provider:
        raise ProviderNotFoundError(f'Provider not found: {provider}')
    return provider

def get_model_and_provider(model    : Union[Model, str], 
                           provider : Union[ProviderType, str, None], 
                           stream   : bool,
                           ignore_working: bool = False,
                           ignore_stream: bool = False,
                           logging: bool = True,
                           has_images: bool = False) -> tuple[str, ProviderType]:
    """
    Retrieves the model and provider based on input parameters.

    Args:
        model (Union[Model, str]): The model to use, either as an object or a string identifier.
        provider (Union[ProviderType, str, None]): The provider to use, either as an object, a string identifier, or None.
        stream (bool): Indicates if the operation should be performed as a stream.
        ignored (list[str], optional): List of provider names to be ignored.
        ignore_working (bool, optional): If True, ignores the working status of the provider.
        ignore_stream (bool, optional): If True, ignores the streaming capability of the provider.

    Returns:
        tuple[str, ProviderType]: A tuple containing the model name and the provider type.

    Raises:
        ProviderNotFoundError: If the provider is not found.
        ModelNotFoundError: If the model is not found.
        ProviderNotWorkingError: If the provider is not working.
        StreamNotSupportedError: If streaming is not supported by the provider.
    """
    if debug.version_check:
        debug.version_check = False
        version.utils.check_version()

    if isinstance(provider, str):
        provider = convert_to_provider(provider)

    if isinstance(model, str):
        if model in ModelUtils.convert:
            model = ModelUtils.convert[model]

    if not provider:
        if not model:
            if has_images:
                model = default_vision
                provider = default_vision.best_provider
            else:
                model = default
                provider = model.best_provider
        elif isinstance(model, str):
            if model in ProviderUtils.convert:
                provider = ProviderUtils.convert[model]
                model = getattr(provider, "default_model", "")
            else:
                raise ModelNotFoundError(f'Model not found: {model}')
        elif isinstance(model, Model):
            provider = model.best_provider
        else:
            raise ValueError(f"Unexpected type: {type(model)}")
    if not provider:
        raise ProviderNotFoundError(f'No provider found for model: {model}')

    provider_name = provider.__name__ if hasattr(provider, "__name__") else type(provider).__name__

    if isinstance(model, Model):
        model = model.name

    if not ignore_working and not provider.working:
        raise ProviderNotWorkingError(f"{provider_name} is not working")

    if isinstance(provider, BaseRetryProvider):
        if not ignore_working:
            provider.providers = [p for p in provider.providers if p.working]

    if not ignore_stream and not provider.supports_stream and stream:
        raise StreamNotSupportedError(f'{provider_name} does not support "stream" argument')

    if logging:
        if model:
            debug.log(f'Using {provider_name} provider and {model} model')
        else:
            debug.log(f'Using {provider_name} provider')

    debug.last_provider = provider
    debug.last_model = model

    return model, provider

def get_last_provider(as_dict: bool = False) -> Union[ProviderType, dict[str, str], None]:
    """
    Retrieves the last used provider.

    Args:
        as_dict (bool, optional): If True, returns the provider information as a dictionary.

    Returns:
        Union[ProviderType, dict[str, str]]: The last used provider, either as an object or a dictionary.
    """
    last = debug.last_provider
    if isinstance(last, BaseRetryProvider):
        last = last.last_provider
    if as_dict:
        if last:
            return {
                "name": last.__name__ if hasattr(last, "__name__") else type(last).__name__,
                "url": last.url,
                "model": debug.last_model,
                "label": getattr(last, "label", None) if hasattr(last, "label") else None
            }
        else:
            return {}
    return last