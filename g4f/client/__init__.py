from __future__ import annotations

import os
import time
import random
import string
import asyncio
import base64
import logging
from typing import Union, AsyncIterator, Iterator, Coroutine, Optional

from ..providers.base_provider import AsyncGeneratorProvider
from ..image import ImageResponse, copy_images, images_dir
from ..typing import Messages, Image, ImageType
from ..providers.types import ProviderType
from ..providers.response import ResponseType, FinishReason, BaseConversation, SynthesizeData
from ..errors import NoImageResponseError, ModelNotFoundError
from ..providers.retry_provider import IterListProvider
from ..providers.base_provider import get_running_loop
from ..Provider.needs_auth.BingCreateImages import BingCreateImages
from .stubs import ChatCompletion, ChatCompletionChunk, Image, ImagesResponse
from .image_models import ImageModels
from .types import IterResponse, ImageProvider, Client as BaseClient
from .service import get_model_and_provider, get_last_provider, convert_to_provider
from .helper import find_stop, filter_json, filter_none, safe_aclose, to_sync_iter, to_async_iterator

ChatCompletionResponseType = Iterator[Union[ChatCompletion, ChatCompletionChunk, BaseConversation]]
AsyncChatCompletionResponseType = AsyncIterator[Union[ChatCompletion, ChatCompletionChunk, BaseConversation]]

try:
    anext # Python 3.8+
except NameError:
    async def anext(aiter):
        try:
            return await aiter.__anext__()
        except StopAsyncIteration:
            raise StopIteration

# Synchronous iter_response function
def iter_response(
    response: Union[Iterator[Union[str, ResponseType]]],
    stream: bool,
    response_format: Optional[dict] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[list[str]] = None
) -> ChatCompletionResponseType:
    content = ""
    finish_reason = None
    completion_id = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
    idx = 0

    if hasattr(response, '__aiter__'):
        # It's an async iterator, wrap it into a sync iterator
        response = to_sync_iter(response)

    for chunk in response:
        if isinstance(chunk, FinishReason):
            finish_reason = chunk.reason
            break
        elif isinstance(chunk, BaseConversation):
            yield chunk
            continue
        elif isinstance(chunk, SynthesizeData):
            continue

        chunk = str(chunk)
        content += chunk

        if max_tokens is not None and idx + 1 >= max_tokens:
            finish_reason = "length"

        first, content, chunk = find_stop(stop, content, chunk if stream else None)

        if first != -1:
            finish_reason = "stop"

        if stream:
            yield ChatCompletionChunk(chunk, None, completion_id, int(time.time()))

        if finish_reason is not None:
            break

        idx += 1

    finish_reason = "stop" if finish_reason is None else finish_reason

    if stream:
        yield ChatCompletionChunk(None, finish_reason, completion_id, int(time.time()))
    else:
        if response_format is not None and "type" in response_format:
            if response_format["type"] == "json_object":
                content = filter_json(content)
        yield ChatCompletion(content, finish_reason, completion_id, int(time.time()))

# Synchronous iter_append_model_and_provider function
def iter_append_model_and_provider(response: ChatCompletionResponseType) -> ChatCompletionResponseType:
    last_provider = None

    for chunk in response:
        if isinstance(chunk, (ChatCompletion, ChatCompletionChunk)):
            last_provider = get_last_provider(True) if last_provider is None else last_provider
            chunk.model = last_provider.get("model")
            chunk.provider = last_provider.get("name")
            yield chunk

async def async_iter_response(
    response: AsyncIterator[Union[str, ResponseType]],
    stream: bool,
    response_format: Optional[dict] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[list[str]] = None
) -> AsyncChatCompletionResponseType:
    content = ""
    finish_reason = None
    completion_id = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
    idx = 0

    try:
        async for chunk in response:
            if isinstance(chunk, FinishReason):
                finish_reason = chunk.reason
                break
            elif isinstance(chunk, BaseConversation):
                yield chunk
                continue
            elif isinstance(chunk, SynthesizeData):
                continue

            chunk = str(chunk)
            content += chunk
            idx += 1

            if max_tokens is not None and idx >= max_tokens:
                finish_reason = "length"

            first, content, chunk = find_stop(stop, content, chunk if stream else None)

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
                    content = filter_json(content)
            yield ChatCompletion(content, finish_reason, completion_id, int(time.time()))
    finally:
        if hasattr(response, 'aclose'):
            await safe_aclose(response)

async def async_iter_append_model_and_provider(
        response: AsyncChatCompletionResponseType
    ) -> AsyncChatCompletionResponseType:
    last_provider = None
    try:
        async for chunk in response:
            if isinstance(chunk, (ChatCompletion, ChatCompletionChunk)):
                last_provider = get_last_provider(True) if last_provider is None else last_provider
                chunk.model = last_provider.get("model")
                chunk.provider = last_provider.get("name")
            yield chunk
    finally:
        if hasattr(response, 'aclose'):
            await safe_aclose(response)

class Client(BaseClient):
    def __init__(
        self,
        provider: Optional[ProviderType] = None,
        image_provider: Optional[ImageProvider] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.chat: Chat = Chat(self, provider)
        self.images: Images = Images(self, image_provider)

class Completions:
    def __init__(self, client: Client, provider: Optional[ProviderType] = None):
        self.client: Client = client
        self.provider: ProviderType = provider

    def create(
        self,
        messages: Messages,
        model: str,
        provider: Optional[ProviderType] = None,
        stream: Optional[bool] = False,
        proxy: Optional[str] = None,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[list[str], str]] = None,
        api_key: Optional[str] = None,
        ignored: Optional[list[str]] = None,
        ignore_working: Optional[bool] = False,
        ignore_stream: Optional[bool] = False,
        **kwargs
    ) -> IterResponse:
        model, provider = get_model_and_provider(
            model,
            self.provider if provider is None else provider,
            stream,
            ignored,
            ignore_working,
            ignore_stream,
        )
        stop = [stop] if isinstance(stop, str) else stop

        response = provider.create_completion(
            model,
            messages,
            stream=stream,
            **filter_none(
                proxy=self.client.proxy if proxy is None else proxy,
                max_tokens=max_tokens,
                stop=stop,
                api_key=self.client.api_key if api_key is None else api_key
            ),
            **kwargs
        )
        if asyncio.iscoroutinefunction(provider.create_completion):
            # Run the asynchronous function in an event loop
            response = asyncio.run(response)
        if stream and hasattr(response, '__aiter__'):
            # It's an async generator, wrap it into a sync iterator
            response = to_sync_iter(response)
        elif hasattr(response, '__aiter__'):
            # If response is an async generator, collect it into a list
            response = list(to_sync_iter(response))
        response = iter_response(response, stream, response_format, max_tokens, stop)
        response = iter_append_model_and_provider(response)
        if stream:
            return response
        else:
            return next(response)

class Chat:
    completions: Completions

    def __init__(self, client: Client, provider: Optional[ProviderType] = None):
        self.completions = Completions(client, provider)

class Images:
    def __init__(self, client: Client, provider: Optional[ProviderType] = None):
        self.client: Client = client
        self.provider: Optional[ProviderType] = provider
        self.models: ImageModels = ImageModels(client)

    def generate(
        self,
        prompt: str,
        model: str = None,
        provider: Optional[ProviderType] = None,
        response_format: str = "url",
        proxy: Optional[str] = None,
        **kwargs
    ) -> ImagesResponse:
        """
        Synchronous generate method that runs the async_generate method in an event loop.
        """
        return asyncio.run(self.async_generate(prompt, model, provider, response_format, proxy, **kwargs))

    async def async_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[ProviderType] = None,
        response_format: Optional[str] = "url",
        proxy: Optional[str] = None,
        **kwargs
    ) -> ImagesResponse:
        if provider is None:
            provider_handler = self.models.get(model, provider or self.provider or BingCreateImages)
        elif isinstance(provider, str):
            provider_handler = convert_to_provider(provider)
        else:
            provider_handler = provider
        if provider_handler is None:
            raise ModelNotFoundError(f"Unknown model: {model}")
        if isinstance(provider_handler, IterListProvider):
            if provider_handler.providers:
                provider_handler = provider_handler.providers[0]
            else:
                raise ModelNotFoundError(f"IterListProvider for model {model} has no providers")
        if proxy is None:
            proxy = self.client.proxy

        response = None
        if isinstance(provider, type) and issubclass(provider, AsyncGeneratorProvider):
            messages = [{"role": "user", "content": f"Generate a image: {prompt}"}]
            async for item in provider_handler.create_async_generator(model, messages, prompt=prompt, **kwargs):
                if isinstance(item, ImageResponse):
                    response = item
                    break
        elif hasattr(provider_handler, 'create'):
            if asyncio.iscoroutinefunction(provider_handler.create):
                response = await provider_handler.create(prompt)
            else:
                response = provider_handler.create(prompt) 
            if isinstance(response, str):
                response = ImageResponse([response], prompt)
        elif hasattr(provider_handler, "create_completion"):
            get_running_loop(check_nested=True)
            messages = [{"role": "user", "content": f"Generate a image: {prompt}"}]
            for item in provider_handler.create_completion(model, messages, prompt=prompt, **kwargs):
                if isinstance(item, ImageResponse):
                    response = item
                    break
        else:
            raise ValueError(f"Provider {provider} does not support image generation")
        if isinstance(response, ImageResponse):
            return await self._process_image_response(
                response,
                response_format,
                proxy,
                model,
                getattr(provider_handler, "__name__", None)
            )
        raise NoImageResponseError(f"Unexpected response type: {type(response)}")

    def create_variation(
        self,
        image: Union[str, bytes],
        model: str = None,
        provider: Optional[ProviderType] = None,
        response_format: str = "url",
        **kwargs
    ) -> ImagesResponse:
        return asyncio.run(self.async_create_variation(
           image, model, provider, response_format, **kwargs
        ))

    async def async_create_variation(
        self,
        image: ImageType,
        model: Optional[str] = None,
        provider: Optional[ProviderType] = None,
        response_format: str = "url",
        proxy: Optional[str] = None,
        **kwargs
    ) -> ImagesResponse:
        if provider is None:
            provider = self.models.get(model, provider or self.provider or BingCreateImages)
            if provider is None:
                raise ModelNotFoundError(f"Unknown model: {model}")
        if isinstance(provider, str):
            provider = convert_to_provider(provider)
        if proxy is None:
            proxy = self.client.proxy

        if isinstance(provider, type) and issubclass(provider, AsyncGeneratorProvider):
            messages = [{"role": "user", "content": "create a variation of this image"}]
            generator = None
            try:
                generator = provider.create_async_generator(model, messages, image=image, response_format=response_format, proxy=proxy, **kwargs)
                async for chunk in generator:
                    if isinstance(chunk, ImageResponse):
                        response = chunk
                        break
            finally:
                if generator and hasattr(generator, 'aclose'):
                    await safe_aclose(generator)
        elif hasattr(provider, 'create_variation'):
            if asyncio.iscoroutinefunction(provider.create_variation):
                response = await provider.create_variation(image, model=model, response_format=response_format, proxy=proxy, **kwargs)
            else:
                response = provider.create_variation(image, model=model, response_format=response_format, proxy=proxy, **kwargs)
        else:
            raise NoImageResponseError(f"Provider {provider} does not support image variation")
    
        if isinstance(response, str):
            response = ImageResponse([response])
        if isinstance(response, ImageResponse):
            return self._process_image_response(response, response_format, proxy, model, getattr(provider, "__name__", None))
        raise NoImageResponseError(f"Unexpected response type: {type(response)}")

    async def _process_image_response(
        self,
        response: ImageResponse,
        response_format: str,
        proxy: str = None,
        model: Optional[str] = None,
        provider: Optional[str] = None
    ) -> list[Image]:
        if response_format in ("url", "b64_json"):
            images = await copy_images(response.get_list(), response.options.get("cookies"), proxy)
            async def process_image_item(image_file: str) -> Image:
                if response_format == "b64_json":
                    with open(os.path.join(images_dir, os.path.basename(image_file)), "rb") as file:
                        image_data = base64.b64encode(file.read()).decode()
                        return Image(url=image_file, b64_json=image_data, revised_prompt=response.alt)
                return Image(url=image_file, revised_prompt=response.alt)
            images = await asyncio.gather(*[process_image_item(image) for image in images])
        else:
            images = [Image(url=image, revised_prompt=response.alt) for image in response.get_list()]
        last_provider = get_last_provider(True)
        return ImagesResponse(
            images,
            model=last_provider.get("model") if model is None else model,
            provider=last_provider.get("name") if provider is None else provider
        )

class AsyncClient(BaseClient):
    def __init__(
        self,
        provider: Optional[ProviderType] = None,
        image_provider: Optional[ImageProvider] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.chat: AsyncChat = AsyncChat(self, provider)
        self.images: AsyncImages = AsyncImages(self, image_provider)

class AsyncChat:
    completions: AsyncCompletions

    def __init__(self, client: AsyncClient, provider: Optional[ProviderType] = None):
        self.completions = AsyncCompletions(client, provider)

class AsyncCompletions:
    def __init__(self, client: AsyncClient, provider: Optional[ProviderType] = None):
        self.client: AsyncClient = client
        self.provider: ProviderType = provider

    def create(
        self,
        messages: Messages,
        model: str,
        provider: Optional[ProviderType] = None,
        stream: Optional[bool] = False,
        proxy: Optional[str] = None,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[list[str], str]] = None,
        api_key: Optional[str] = None,
        ignored: Optional[list[str]] = None,
        ignore_working: Optional[bool] = False,
        ignore_stream: Optional[bool] = False,
        **kwargs
    ) -> Union[Coroutine[ChatCompletion], AsyncIterator[ChatCompletionChunk, BaseConversation]]:
        model, provider = get_model_and_provider(
            model,
            self.provider if provider is None else provider,
            stream,
            ignored,
            ignore_working,
            ignore_stream,
        )
        stop = [stop] if isinstance(stop, str) else stop

        response = provider.create_completion(
            model,
            messages,
            stream=stream,
            **filter_none(
                proxy=self.client.proxy if proxy is None else proxy,
                max_tokens=max_tokens,
                stop=stop,
                api_key=self.client.api_key if api_key is None else api_key
            ),
            **kwargs
        )

        if not isinstance(response, AsyncIterator):
            response = to_async_iterator(response)
        response = async_iter_response(response, stream, response_format, max_tokens, stop)
        response = async_iter_append_model_and_provider(response)
        return response if stream else anext(response)

class AsyncImages(Images):
    def __init__(self, client: AsyncClient, provider: Optional[ProviderType] = None):
        self.client: AsyncClient = client
        self.provider: Optional[ProviderType] = provider
        self.models: ImageModels = ImageModels(client)

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[ProviderType] = None,
        response_format: str = "url",
        **kwargs
    ) -> ImagesResponse:
        return await self.async_generate(prompt, model, provider, response_format, **kwargs)

    async def create_variation(
        self,
        image: ImageType,
        model: str = None,
        provider: ProviderType = None,
        response_format: str = "url",
        **kwargs
    ) -> ImagesResponse:
        return await self.async_create_variation(
           image, model, provider, response_format, **kwargs
        )