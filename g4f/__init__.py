from __future__ import annotations

import os
import logging
from typing import Union, Optional, Coroutine

from . import debug, version
from .models import Model
from .client import Client, AsyncClient
from .typing import Messages, CreateResult, AsyncResult, ImageType
from .cookies import get_cookies, set_cookies
from .providers.types import ProviderType
from .providers.helper import concat_chunks, async_concat_chunks
from .client.service import get_model_and_provider

# Configure logger
logger = logging.getLogger("g4f")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
logger.addHandler(handler)
logger.setLevel(logging.ERROR)


class ChatCompletion:
    @staticmethod
    def _prepare_request(model: Union[Model, str],
                         messages: Messages,
                         provider: Union[ProviderType, str, None],
                         stream: bool,
                         image: ImageType,
                         image_name: Optional[str],
                         ignore_working: bool,
                         ignore_stream: bool,
                         **kwargs):
        """Shared pre-processing for sync/async create methods."""
        if image is not None:
            kwargs["media"] = [(image, image_name)]
        elif "images" in kwargs:
            kwargs["media"] = kwargs.pop("images")

        model, provider = get_model_and_provider(
            model, provider, stream,
            ignore_working,
            ignore_stream,
            has_images="media" in kwargs,
        )

        if "proxy" not in kwargs:
            proxy = os.environ.get("G4F_PROXY")
            if proxy:
                kwargs["proxy"] = proxy
        if ignore_stream:
            kwargs["ignore_stream"] = True

        return model, provider, kwargs

    @staticmethod
    def create(model: Union[Model, str],
               messages: Messages,
               provider: Union[ProviderType, str, None] = None,
               stream: bool = False,
               image: ImageType = None,
               image_name: Optional[str] = None,
               ignore_working: bool = False,
               ignore_stream: bool = False,
               **kwargs) -> Union[CreateResult, str]:
        model, provider, kwargs = ChatCompletion._prepare_request(
            model, messages, provider, stream, image, image_name,
            ignore_working, ignore_stream, **kwargs
        )
        result = provider.create_function(model, messages, stream=stream, **kwargs)
        return result if stream or ignore_stream else concat_chunks(result)

    @staticmethod
    def create_async(model: Union[Model, str],
                     messages: Messages,
                     provider: Union[ProviderType, str, None] = None,
                     stream: bool = False,
                     image: ImageType = None,
                     image_name: Optional[str] = None,
                     ignore_working: bool = False,
                     ignore_stream: bool = False,
                     **kwargs) -> Union[AsyncResult, Coroutine[str]]:
        model, provider, kwargs = ChatCompletion._prepare_request(
            model, messages, provider, stream, image, image_name,
            ignore_working, ignore_stream, **kwargs
        )
        result = provider.async_create_function(model, messages, stream=stream, **kwargs)
        if not stream and not ignore_stream and hasattr(result, "__aiter__"):
            result = async_concat_chunks(result)
        return result