from __future__ import annotations

import re
import os
import time
import random
import string

from .types import Client as BaseClient
from .types import BaseProvider, ProviderType, FinishReason
from .stubs import ChatCompletion, ChatCompletionChunk, Image, ImagesResponse
from ..typing import Union, Iterator, Messages, ImageType, AsyncIerator
from ..image import ImageResponse as ImageProviderResponse
from ..errors import NoImageResponseError, RateLimitError, MissingAuthError
from .. import get_model_and_provider, get_last_provider
from .helper import read_json, find_stop, filter_none
ä
async def iter_response(
    response: AsyncIterator[str],
    stream: bool,
    response_format: dict = None,
    max_tokens: int = None,
    stop: list = None
) -> AsyncIterResponse:
    content = ""
    finish_reason = None
    completion_id = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
    count: int = 0
    async for chunk in response:
        if isinstance(chunk, FinishReason):
            finish_reason = chunk.reason
            break
        content += str(chunk)
        count += 1
        if max_tokens is not None and count >= max_tokens:
            finish_reason = "length"
        first, content, chunk = find_stop(stop, content, chunk)
        if first != -1:
            finish_reason = "stop"
        if stream:
            yield ChatCompletionChunk(chunk, None, completion_id, int(time.time()))
        if finish_reason is not None:
            break
    finish_reason = "stop" if finish_reason is None else finish_reason
    if stream:
        yield ChatCompletionChunk(None, finish_reason, completion_id, int(time.time()))
    else:
        if response_format is not None and "type" in response_format:
            if response_format["type"] == "json_object":
                content = read_json(content)
        yield ChatCompletion(content, finish_reason, completion_id, int(time.time()))

async def iter_append_model_and_provider(response: AsyncIterResponse) -> IterResponse:
    last_provider = None
    async for chunk in response:
        last_provider = get_last_provider(True) if last_provider is None else last_provider
        chunk.model = last_provider.get("model")
        chunk.provider =  last_provider.get("name")
        yield chunk

class Client(BaseClient):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.chat: Chat = Chat(self, provider)
        self.images: Images = Images(self, image_provider)

async def cast_iter_async(iter):
    for chunk in iter:
        yield chunk

def create_response(
    messages: Messages,
    model: str,
    provider: ProviderType = None,
    stream: bool = False,
    response_format: dict = None,
    max_tokens: int = None,
    stop: Union[list[str], str] = None,
    api_key: str = None,
    **kwargs
):
    if hasattr(provider, "create_async_generator):
        create = provider.create_async_generator
    else:
        create = provider.create_completion
    response = create(
        model, messages, stream,            
        **filter_none(
            proxy=self.client.get_proxy(),
            max_tokens=max_tokens,
            stop=stop,
            api_key=self.client.api_key if api_key is None else api_key
        ),
        **kwargs
    )
    if not hasattr(provider, "create_async_generator")
        response = cast_iter_async(response)
    return response

class Completions():
    def __init__(self, client: Client, provider: ProviderType = None):
        self.client: Client = client
        self.provider: ProviderType = provider

    def create(
        self,
        messages: Messages,
        model: str,
        provider: ProviderType = None,
        stream: bool = False,
        response_format: dict = None,
        ignored  : list[str] = None,
        ignore_working: bool = False,
        ignore_stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        model, provider = get_model_and_provider(
            model,
            self.provider if provider is None else provider,
            stream,
            ignored,
            ignore_working,
            ignore_stream,
            **kwargs
        )
        stop = [stop] if isinstance(stop, str) else stop
        response = create_response(messages, model, provider, stream, **kwargs)
        response = iter_response(response, stream, response_format, max_tokens, stop)
        response = iter_append_model_and_provider(response)
        return response if stream else anext(response)

class Chat():
    completions: Completions

    def __init__(self, client: Client, provider: ProviderType = None):
        self.completions = Completions(client, provider)

async def iter_image_response(response: Iterator) -> Union[ImagesResponse, None]:
    async for chunk in list(response):
        if isinstance(chunk, ImageProviderResponse):
            return ImagesResponse([Image(image) for image in chunk.get_list()])

def create_image(client: Client, provider: ProviderType, prompt: str, model: str = "", **kwargs) -> AsyncIterator:
    prompt = f"create a image with: {prompt}"
    return provider.create_async_generator(
        model,
        [{"role": "user", "content": prompt}],
        True,
        proxy=client.get_proxy(),
        **kwargs
    )

class Images():
    def __init__(self, client: Client, provider: ImageProvider = None):
        self.client: Client = client
        self.provider: ImageProvider = provider
        self.models: ImageModels = ImageModels(client)

    async def generate(self, prompt, model: str = None, **kwargs) -> ImagesResponse:
        provider = self.models.get(model, self.provider)
        if isinstance(provider, type) and issubclass(provider, BaseProvider):
            response = create_image(self.client, provider, prompt, **kwargs)
        else:
            try:
                response = list(provider.create(prompt))
            except (RateLimitError, MissingAuthError) as e:
                # Fallback for default provider
                if self.provider is None:
                    response = create_image(self.client, self.models.you, prompt, model or "dall-e", **kwargs)
                else:
                    raise e
        image = iter_image_response(response)
        if image is None:
            raise NoImageResponseError()
        return image

    async def create_variation(self, image: ImageType, model: str = None, **kwargs):
        provider = self.models.get(model, self.provider)
        result = None
        if isinstance(provider, type) and issubclass(provider, BaseProvider):
            response = provider.create_async_generator(
                "",
                [{"role": "user", "content": "create a image like this"}],
                True,
                image=image,
                proxy=self.client.get_proxy(),
                **kwargs
            )
            async for chunk in response:
                if isinstance(chunk, ImageProviderResponse):
                    result = ([chunk.images] if isinstance(chunk.images, str) else chunk.images)
                    result = ImagesResponse([Image(image)for image in result])
        if result is None:
            raise NoImageResponseError()
        return result
