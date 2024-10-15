from __future__ import annotations

import os
import time
import random
import string
import threading
import asyncio
import base64
import aiohttp
import queue
from typing import Union, AsyncIterator, Iterator

from ..providers.base_provider import AsyncGeneratorProvider
from ..image import ImageResponse, to_image, to_data_uri
from ..typing import Messages, ImageType
from ..providers.types import BaseProvider, ProviderType, FinishReason
from ..providers.conversation import BaseConversation
from ..image import ImageResponse as ImageProviderResponse
from ..errors import NoImageResponseError
from .stubs import ChatCompletion, ChatCompletionChunk, Image, ImagesResponse
from .image_models import ImageModels
from .types import IterResponse, ImageProvider
from .types import Client as BaseClient
from .service import get_model_and_provider, get_last_provider
from .helper import find_stop, filter_json, filter_none
from ..models import ModelUtils
from ..Provider import IterListProvider

# Helper function to convert an async generator to a synchronous iterator
def to_sync_iter(async_gen: AsyncIterator) -> Iterator:
    q = queue.Queue()
    loop = asyncio.new_event_loop()
    done = object()

    def _run():
        asyncio.set_event_loop(loop)

        async def iterate():
            try:
                async for item in async_gen:
                    q.put(item)
            finally:
                q.put(done)

        loop.run_until_complete(iterate())
        loop.close()

    threading.Thread(target=_run).start()

    while True:
        item = q.get()
        if item is done:
            break
        yield item

# Helper function to convert a synchronous iterator to an async iterator
async def to_async_iterator(iterator):
    for item in iterator:
        yield item

# Synchronous iter_response function
def iter_response(
    response: Union[Iterator[str], AsyncIterator[str]],
    stream: bool,
    response_format: dict = None,
    max_tokens: int = None,
    stop: list = None
) -> Iterator[Union[ChatCompletion, ChatCompletionChunk]]:
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

        content += str(chunk)

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
def iter_append_model_and_provider(response: Iterator) -> Iterator:
    last_provider = None

    for chunk in response:
        last_provider = get_last_provider(True) if last_provider is None else last_provider
        chunk.model = last_provider.get("model")
        chunk.provider = last_provider.get("name")
        yield chunk

class Client(BaseClient):
    def __init__(
        self,
        provider: ProviderType = None,
        image_provider: ImageProvider = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.chat: Chat = Chat(self, provider)
        self._images: Images = Images(self, image_provider)

    @property
    def images(self) -> Images:
        return self._images

    async def async_images(self) -> Images:
        return self._images

class Completions:
    def __init__(self, client: Client, provider: ProviderType = None):
        self.client: Client = client
        self.provider: ProviderType = provider

    def create(
        self,
        messages: Messages,
        model: str,
        provider: ProviderType = None,
        stream: bool = False,
        proxy: str = None,
        response_format: dict = None,
        max_tokens: int = None,
        stop: Union[list[str], str] = None,
        api_key: str = None,
        ignored: list[str] = None,
        ignore_working: bool = False,
        ignore_stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        model, provider = get_model_and_provider(
            model,
            self.provider if provider is None else provider,
            stream,
            ignored,
            ignore_working,
            ignore_stream,
        )

        stop = [stop] if isinstance(stop, str) else stop

        if asyncio.iscoroutinefunction(provider.create_completion):
            # Run the asynchronous function in an event loop
            response = asyncio.run(provider.create_completion(
                model,
                messages,
                stream=stream,
                **filter_none(
                    proxy=self.client.get_proxy() if proxy is None else proxy,
                    max_tokens=max_tokens,
                    stop=stop,
                    api_key=self.client.api_key if api_key is None else api_key
                ),
                **kwargs
            ))
        else:
            response = provider.create_completion(
                model,
                messages,
                stream=stream,
                **filter_none(
                    proxy=self.client.get_proxy() if proxy is None else proxy,
                    max_tokens=max_tokens,
                    stop=stop,
                    api_key=self.client.api_key if api_key is None else api_key
                ),
                **kwargs
            )

        if stream:
            if hasattr(response, '__aiter__'):
                # It's an async generator, wrap it into a sync iterator
                response = to_sync_iter(response)

            # Now 'response' is an iterator
            response = iter_response(response, stream, response_format, max_tokens, stop)
            response = iter_append_model_and_provider(response)
            return response
        else:
            if hasattr(response, '__aiter__'):
                # If response is an async generator, collect it into a list
                response = list(to_sync_iter(response))
            response = iter_response(response, stream, response_format, max_tokens, stop)
            response = iter_append_model_and_provider(response)
            return next(response)

    async def async_create(
        self,
        messages: Messages,
        model: str,
        provider: ProviderType = None,
        stream: bool = False,
        proxy: str = None,
        response_format: dict = None,
        max_tokens: int = None,
        stop: Union[list[str], str] = None,
        api_key: str = None,
        ignored: list[str] = None,
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
        )

        stop = [stop] if isinstance(stop, str) else stop

        if asyncio.iscoroutinefunction(provider.create_completion):
            response = await provider.create_completion(
                model,
                messages,
                stream=stream,
                **filter_none(
                    proxy=self.client.get_proxy() if proxy is None else proxy,
                    max_tokens=max_tokens,
                    stop=stop,
                    api_key=self.client.api_key if api_key is None else api_key
                ),
                **kwargs
            )
        else:
            response = provider.create_completion(
                model,
                messages,
                stream=stream,
                **filter_none(
                    proxy=self.client.get_proxy() if proxy is None else proxy,
                    max_tokens=max_tokens,
                    stop=stop,
                    api_key=self.client.api_key if api_key is None else api_key
                ),
                **kwargs
            )

        # Removed 'await' here since 'async_iter_response' returns an async generator
        response = async_iter_response(response, stream, response_format, max_tokens, stop)
        response = async_iter_append_model_and_provider(response)

        if stream:
            return response
        else:
            async for result in response:
                return result

class Chat:
    completions: Completions

    def __init__(self, client: Client, provider: ProviderType = None):
        self.completions = Completions(client, provider)

# Asynchronous versions of the helper functions
async def async_iter_response(
    response: Union[AsyncIterator[str], Iterator[str]],
    stream: bool,
    response_format: dict = None,
    max_tokens: int = None,
    stop: list = None
) -> AsyncIterator[Union[ChatCompletion, ChatCompletionChunk]]:
    content = ""
    finish_reason = None
    completion_id = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
    idx = 0

    if not hasattr(response, '__aiter__'):
        response = to_async_iterator(response)

    async for chunk in response:
        if isinstance(chunk, FinishReason):
            finish_reason = chunk.reason
            break
        elif isinstance(chunk, BaseConversation):
            yield chunk
            continue

        content += str(chunk)

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

async def async_iter_append_model_and_provider(response: AsyncIterator) -> AsyncIterator:
    last_provider = None

    if not hasattr(response, '__aiter__'):
        response = to_async_iterator(response)

    async for chunk in response:
        last_provider = get_last_provider(True) if last_provider is None else last_provider
        chunk.model = last_provider.get("model")
        chunk.provider = last_provider.get("name")
        yield chunk

async def iter_image_response(response: AsyncIterator) -> Union[ImagesResponse, None]:
    response_list = []
    async for chunk in response:
        if isinstance(chunk, ImageProviderResponse):
            response_list.extend(chunk.get_list())
        elif isinstance(chunk, str):
            response_list.append(chunk)

    if response_list:
        return ImagesResponse([Image(image) for image in response_list])

    return None

async def create_image(client: Client, provider: ProviderType, prompt: str, model: str = "", **kwargs) -> AsyncIterator:
    if isinstance(provider, type) and provider.__name__ == "You":
        kwargs["chat_mode"] = "create"
    else:
        prompt = f"create an image with: {prompt}"

    if asyncio.iscoroutinefunction(provider.create_completion):
        response = await provider.create_completion(
            model,
            [{"role": "user", "content": prompt}],
            stream=True,
            proxy=client.get_proxy(),
            **kwargs
        )
    else:
        response = provider.create_completion(
            model,
            [{"role": "user", "content": prompt}],
            stream=True,
            proxy=client.get_proxy(),
            **kwargs
        )

    # Wrap synchronous iterator into async iterator if necessary
    if not hasattr(response, '__aiter__'):
        response = to_async_iterator(response)

    return response

class Image:
    def __init__(self, url: str = None, b64_json: str = None):
        self.url = url
        self.b64_json = b64_json

    def __repr__(self):
        return f"Image(url={self.url}, b64_json={'<base64 data>' if self.b64_json else None})"

class ImagesResponse:
    def __init__(self, data: list[Image]):
        self.data = data

    def __repr__(self):
        return f"ImagesResponse(data={self.data})"

class Images:
    def __init__(self, client: 'Client', provider: 'ImageProvider' = None):
        self.client: 'Client' = client
        self.provider: 'ImageProvider' = provider
        self.models: ImageModels = ImageModels(client)

    def generate(self, prompt: str, model: str = None, response_format: str = "url", **kwargs) -> ImagesResponse:
        """
        Synchronous generate method that runs the async_generate method in an event loop.
        """
        return asyncio.run(self.async_generate(prompt, model, response_format=response_format, **kwargs))

    async def async_generate(self, prompt: str, model: str = None, response_format: str = "url", **kwargs) -> ImagesResponse:
        provider = self.models.get(model, self.provider)
        if provider is None:
            raise ValueError(f"Unknown model: {model}")

        if isinstance(provider, IterListProvider):
            if provider.providers:
                provider = provider.providers[0]
            else:
                raise ValueError(f"IterListProvider for model {model} has no providers")

        if isinstance(provider, type) and issubclass(provider, AsyncGeneratorProvider):
            messages = [{"role": "user", "content": prompt}]
            async for response in provider.create_async_generator(model, messages, **kwargs):
                if isinstance(response, ImageResponse):
                    return await self._process_image_response(response, response_format)
                elif isinstance(response, str):
                    image_response = ImageResponse([response], prompt)
                    return await self._process_image_response(image_response, response_format)
        elif hasattr(provider, 'create'):
            if asyncio.iscoroutinefunction(provider.create):
                response = await provider.create(prompt)
            else:
                response = provider.create(prompt)

            if isinstance(response, ImageResponse):
                return await self._process_image_response(response, response_format)
            elif isinstance(response, str):
                image_response = ImageResponse([response], prompt)
                return await self._process_image_response(image_response, response_format)
        else:
            raise ValueError(f"Provider {provider} does not support image generation")

        raise NoImageResponseError(f"Unexpected response type: {type(response)}")

    async def _process_image_response(self, response: ImageResponse, response_format: str) -> ImagesResponse:
        processed_images = []

        for image_data in response.get_list():
            if image_data.startswith('http://') or image_data.startswith('https://'):
                if response_format == "url":
                    processed_images.append(Image(url=image_data))
                elif response_format == "b64_json":
                    # Fetch the image data and convert it to base64
                    image_content = await self._fetch_image(image_data)
                    b64_json = base64.b64encode(image_content).decode('utf-8')
                    processed_images.append(Image(b64_json=b64_json))
            else:
                # Assume image_data is base64 data or binary
                if response_format == "url":
                    if image_data.startswith('data:image'):
                        # Remove the data URL scheme and get the base64 data
                        header, base64_data = image_data.split(',', 1)
                    else:
                        base64_data = image_data
                    # Decode the base64 data
                    image_data_bytes = base64.b64decode(base64_data)
                    # Convert bytes to an image
                    image = to_image(image_data_bytes)
                    file_name = self._save_image(image)
                    processed_images.append(Image(url=file_name))
                elif response_format == "b64_json":
                    if isinstance(image_data, bytes):
                        b64_json = base64.b64encode(image_data).decode('utf-8')
                    else:
                        b64_json = image_data  # If already base64-encoded string
                    processed_images.append(Image(b64_json=b64_json))

        return ImagesResponse(processed_images)

    async def _fetch_image(self, url: str) -> bytes:
        # Asynchronously fetch image data from the URL
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    raise Exception(f"Failed to fetch image from {url}, status code {resp.status}")

    def _save_image(self, image: 'PILImage') -> str:
        os.makedirs('generated_images', exist_ok=True)
        file_name = f"generated_images/image_{int(time.time())}_{random.randint(0, 10000)}.png"
        image.save(file_name)
        return file_name

    async def create_variation(self, image: Union[str, bytes], model: str = None, response_format: str = "url", **kwargs):
        # Existing implementation, adjust if you want to support b64_json here as well
        pass
