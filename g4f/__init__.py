from __future__ import annotations

import os
import logging
from typing import Union, Optional, Coroutine

from . import debug, version
from .models import Model
from .client import Client, AsyncClient
from .typing import Messages, CreateResult, AsyncResult, ImageType
from .errors import StreamNotSupportedError
from .cookies import get_cookies, set_cookies
from .providers.types import ProviderType
from .providers.helper import concat_chunks
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
               ignore_working: bool = False,
               ignore_stream: bool = False,
               **kwargs) -> Union[CreateResult, str]:
        model, provider = get_model_and_provider(
            model, provider, stream,
            ignore_working,
            ignore_stream
        )
        if image is not None:
            kwargs["images"] = [(image, image_name)]
        if "proxy" not in kwargs:
            proxy = os.environ.get("G4F_PROXY")
            if proxy:
                kwargs["proxy"] = proxy
        if ignore_stream:
            kwargs["ignore_stream"] = True

        result = provider.create_completion(model, messages, stream=stream, **kwargs)

        return result if stream else concat_chunks(result)

    @staticmethod
    def create_async(model    : Union[Model, str],
                     messages : Messages,
                     provider : Union[ProviderType, str, None] = None,
                     stream   : bool = False,
                     image    : ImageType = None,
                     image_name: Optional[str] = None,
                     ignore_stream: bool = False,
                     ignore_working: bool = False,
                     **kwargs) -> Union[AsyncResult, Coroutine[str]]:
        model, provider = get_model_and_provider(model, provider, False, ignore_working)
        if image is not None:
            kwargs["images"] = [(image, image_name)]
        if "proxy" not in kwargs:
            proxy = os.environ.get("G4F_PROXY")
            if proxy:
                kwargs["proxy"] = proxy
        if ignore_stream:
            kwargs["ignore_stream"] = True

        if stream:
            if hasattr(provider, "create_async_generator"):
                return provider.create_async_generator(model, messages, **kwargs)
            raise StreamNotSupportedError(f'{provider.__name__} does not support "stream" argument in "create_async"')

        return provider.create_async(model, messages, **kwargs)