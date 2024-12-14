from __future__ import annotations

import os
import logging
from typing import Union, Optional

from . import debug, version
from .models import Model
from .client import Client, AsyncClient
from .typing import Messages, CreateResult, AsyncResult, ImageType
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
               image    : ImageType = None,
               image_name: Optional[str] = None,
               ignored: list[str] = None, 
               ignore_working: bool = False,
               ignore_stream: bool = False,
               **kwargs) -> Union[CreateResult, str]:
        model, provider = get_model_and_provider(
            model, provider, stream,
            ignored, ignore_working,
            ignore_stream or kwargs.get("ignore_stream_and_auth")
        )
        if image is not None:
            kwargs["images"] = [(image, image_name)]
        if "proxy" not in kwargs:
            proxy = os.environ.get("G4F_PROXY")
            if proxy:
                kwargs['proxy'] = proxy

        result = provider.create_completion(model, messages, stream=stream, **kwargs)

        return result if stream else ''.join([str(chunk) for chunk in result if chunk])

    @staticmethod
    def create_async(model    : Union[Model, str],
                     messages : Messages,
                     provider : Union[ProviderType, str, None] = None,
                     stream   : bool = False,
                     ignored  : list[str] = None,
                     ignore_working: bool = False,
                     **kwargs) -> Union[AsyncResult, str]:
        model, provider = get_model_and_provider(model, provider, False, ignored, ignore_working)

        if stream:
            if hasattr(provider, "create_async_generator"):
                return provider.create_async_generator(model, messages, **kwargs)
            raise StreamNotSupportedError(f'{provider.__name__} does not support "stream" argument in "create_async"')

        return provider.create_async(model, messages, **kwargs)