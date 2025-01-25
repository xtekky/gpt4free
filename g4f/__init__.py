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
from .providers.helper import concat_chunks, async_concat_chunks
from .client.service import get_model_and_provider

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
        if image is not None:
            kwargs["images"] = [(image, image_name)]
        model, provider = get_model_and_provider(
            model, provider, stream,
            ignore_working,
            ignore_stream,
            has_images="images" in kwargs,
        )
        if "proxy" not in kwargs:
            proxy = os.environ.get("G4F_PROXY")
            if proxy:
                kwargs["proxy"] = proxy
        if ignore_stream:
            kwargs["ignore_stream"] = True

        result = provider.get_create_function()(model, messages, stream=stream, **kwargs)

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
        if image is not None:
            kwargs["images"] = [(image, image_name)]
        model, provider = get_model_and_provider(model, provider, False, ignore_working, has_images="images" in kwargs)
        if "proxy" not in kwargs:
            proxy = os.environ.get("G4F_PROXY")
            if proxy:
                kwargs["proxy"] = proxy
        if ignore_stream:
            kwargs["ignore_stream"] = True

        result = provider.get_async_create_function()(model, messages, stream=stream, **kwargs)

        if not stream:
            if hasattr(result, "__aiter__"):
                result = async_concat_chunks(result)

        return result