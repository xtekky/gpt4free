from __future__ import annotations

import os
import time
import random
import string
import asyncio
import base64
import aiohttp
import logging
from typing import Union, AsyncIterator, Iterator, Coroutine

from ..providers.base_provider import AsyncGeneratorProvider
from ..image import ImageResponse, to_image, to_data_uri, is_accepted_format, EXTENSIONS_MAP
from ..typing import Messages, Cookies, Image
from ..providers.types import ProviderType, FinishReason, BaseConversation
from ..errors import NoImageResponseError
from ..providers.retry_provider import IterListProvider
from ..Provider.needs_auth.BingCreateImages import BingCreateImages
from ..requests.aiohttp import get_connector
from .stubs import ChatCompletion, ChatCompletionChunk, Image, ImagesResponse
from .image_models import ImageModels
from .types import IterResponse, ImageProvider, Client as BaseClient
from .service import get_model_and_provider, get_last_provider, convert_to_provider
from .helper import find_stop, filter_json, filter_none, safe_aclose, to_sync_iter, to_async_iterator

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
def iter_append_model_and_provider(response: Iterator[ChatCompletionChunk]) -> Iterator[ChatCompletionChunk]:
    last_provider = None

    for chunk in response:
        last_provider = get_last_provider(True) if last_provider is None else last_provider
        chunk.model = last_provider.get("model")
        chunk.provider = last_provider.get("name")
        yield chunk

async def async_iter_response(
    response: AsyncIterator[str],
    stream: bool,
    response_format: dict = None,
    max_tokens: int = None,
    stop: list = None
) -> AsyncIterator[Union[ChatCompletion, ChatCompletionChunk]]:
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

async def async_iter_append_model_and_provider(response: AsyncIterator[ChatCompletionChunk]) -> AsyncIterator:
    last_provider = None
    try:
        async for chunk in response:
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
        provider: ProviderType = None,
        image_provider: ImageProvider = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.chat: Chat = Chat(self, provider)
        self.images: Images = Images(self, image_provider)

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

    def __init__(self, client: Client, provider: ProviderType = None):
        self.completions = Completions(client, provider)

class Images:
    def __init__(self, client: Client, provider: ProviderType = None):
        self.client: Client = client
        self.provider: ProviderType = provider
        self.models: ImageModels = ImageModels(client)

    def generate(self, prompt: str, model: str = None, provider: ProviderType = None, response_format: str = "url", proxy: str = None, **kwargs) -> ImagesResponse:
        """
        Synchronous generate method that runs the async_generate method in an event loop.
        """
        return asyncio.run(self.async_generate(prompt, model, provider, response_format=response_format, proxy=proxy, **kwargs))

    async def async_generate(self, prompt: str, model: str = None, provider: ProviderType = None, response_format: str = "url", proxy: str = None, **kwargs) -> ImagesResponse:
        if provider is None:
            provider_handler = self.models.get(model, provider or self.provider or BingCreateImages)
        elif isinstance(provider, str):
            provider_handler = convert_to_provider(provider)
        if provider_handler is None:
            raise ValueError(f"Unknown model: {model}")
        if proxy is None:
            proxy = self.client.proxy

        if isinstance(provider_handler, IterListProvider):
            if provider_handler.providers:
                provider_handler = provider_handler.providers[0]
            else:
                raise ValueError(f"IterListProvider for model {model} has no providers")

        response = None
        if hasattr(provider_handler, "create_async_generator"):
            messages = [{"role": "user", "content": prompt}]
            async for item in provider_handler.create_async_generator(model, messages, **kwargs):
                if isinstance(item, ImageResponse):
                    response = item
                    break
        elif hasattr(provider, 'create'):
            if asyncio.iscoroutinefunction(provider_handler.create):
                response = await provider_handler.create(prompt)
            else:
                response = provider_handler.create(prompt) 
            if isinstance(response, str):
                response = ImageResponse([response], prompt)
        else:
            raise ValueError(f"Provider {provider} does not support image generation")
        if isinstance(response, ImageResponse):
            return await self._process_image_response(response, response_format, proxy, model=model, provider=provider)

        raise NoImageResponseError(f"Unexpected response type: {type(response)}")

    async def _process_image_response(self, response: ImageResponse, response_format: str, proxy: str = None, model: str = None, provider: str = None) -> ImagesResponse:
            async def process_image_item(session: aiohttp.ClientSession, image_data: str):
                image_data_bytes = None
                if image_data.startswith("http://") or image_data.startswith("https://"):
                    if response_format == "url":
                        return Image(url=image_data, revised_prompt=response.alt)
                    elif response_format == "b64_json":
                        # Fetch the image data and convert it to base64
                        image_data_bytes = await self._fetch_image(session, image_data)
                        b64_json = base64.b64encode(image_data_bytes).decode("utf-8")
                        return Image(b64_json=b64_json, url=image_data, revised_prompt=response.alt)
                else:
                    # Assume image_data is base64 data or binary
                    if response_format == "url":
                        if image_data.startswith("data:image"):
                            # Remove the data URL scheme and get the base64 data
                            base64_data = image_data.split(",", 1)[-1]
                        else:
                            base64_data = image_data
                        # Decode the base64 data
                        image_data_bytes = base64.b64decode(base64_data)
                if image_data_bytes:
                    file_name = self._save_image(image_data_bytes)
                    return Image(url=file_name, revised_prompt=response.alt)
                else:
                    raise ValueError("Unable to process image data")

            last_provider = get_last_provider(True)
            async with aiohttp.ClientSession(cookies=response.get("cookies"), connector=get_connector(proxy=proxy)) as session:
                return ImagesResponse(
                    await asyncio.gather(*[process_image_item(session, image_data) for image_data in response.get_list()]),
                    model=last_provider.get("model") if model is None else model,
                    provider=last_provider.get("name") if provider is None else provider
                )

    async def _fetch_image(self, session: aiohttp.ClientSession, url: str) -> bytes:
        # Asynchronously fetch image data from the URL
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                raise RuntimeError(f"Failed to fetch image from {url}, status code {resp.status}")

    def _save_image(self, image_data_bytes: bytes) -> str:
        os.makedirs('generated_images', exist_ok=True)
        image = to_image(image_data_bytes)
        file_name = f"generated_images/image_{int(time.time())}_{random.randint(0, 10000)}.{EXTENSIONS_MAP[is_accepted_format(image_data_bytes)]}"
        image.save(file_name)
        return file_name

    def create_variation(self, image: Union[str, bytes], model: str = None, provider: ProviderType = None, response_format: str = "url", **kwargs) -> ImagesResponse:
        return asyncio.run(self.async_create_variation(
           image, model, provider, response_format, **kwargs
        ))

    async def async_create_variation(self, image: Union[str, bytes], model: str = None, provider: ProviderType = None, response_format: str = "url", proxy: str = None, **kwargs) -> ImagesResponse:
        if provider is None:
            provider = self.models.get(model, provider or self.provider or BingCreateImages)
            if provider is None:
                raise ValueError(f"Unknown model: {model}")
        if isinstance(provider, str):
            provider = convert_to_provider(provider)
        if proxy is None:
            proxy = self.client.proxy

        if isinstance(provider, type) and issubclass(provider, AsyncGeneratorProvider):
            messages = [{"role": "user", "content": "create a variation of this image"}]
            image_data = to_data_uri(image)
            generator = None
            try:
                generator = provider.create_async_generator(model, messages, image=image_data, response_format=response_format, proxy=proxy, **kwargs)
                async for response in generator:
                    if isinstance(response, ImageResponse):
                        return self._process_image_response(response)
            except RuntimeError as e:
                if "async generator ignored GeneratorExit" in str(e):
                    logging.warning("Generator ignored GeneratorExit in create_variation, handling gracefully")
                else:
                    raise
            finally:
                if generator and hasattr(generator, 'aclose'):
                    await safe_aclose(generator)
                logging.info("AsyncGeneratorProvider processing completed in create_variation")
        elif hasattr(provider, 'create_variation'):
            if asyncio.iscoroutinefunction(provider.create_variation):
                response = await provider.create_variation(image, model=model, response_format=response_format, proxy=proxy, **kwargs)
            else:
                response = provider.create_variation(image, model=model, response_format=response_format, proxy=proxy, **kwargs)
            if isinstance(response, str):
                response = ImageResponse([response])
            return self._process_image_response(response)
        else:
            raise ValueError(f"Provider {provider} does not support image variation")

class AsyncClient(BaseClient):
    def __init__(
        self,
        provider: ProviderType = None,
        image_provider: ImageProvider = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.chat: AsyncChat = AsyncChat(self, provider)
        self.images: AsyncImages = AsyncImages(self, image_provider)

class AsyncChat:
    completions: AsyncCompletions

    def __init__(self, client: AsyncClient, provider: ProviderType = None):
        self.completions = AsyncCompletions(client, provider)

class AsyncCompletions:
    def __init__(self, client: AsyncClient, provider: ProviderType = None):
        self.client: AsyncClient = client
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
    ) -> Union[Coroutine[ChatCompletion], AsyncIterator[ChatCompletionChunk]]:
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
    def __init__(self, client: AsyncClient, provider: ImageProvider = None):
        self.client: AsyncClient = client
        self.provider: ImageProvider = provider
        self.models: ImageModels = ImageModels(client)

    async def generate(self, prompt: str, model: str = None, provider: ProviderType = None, response_format: str = "url", **kwargs) -> ImagesResponse:
        return await self.async_generate(prompt, model, provider, response_format, **kwargs)

    async def create_variation(self, image: Union[str, bytes], model: str = None, provider: ProviderType = None, response_format: str = "url", **kwargs) -> ImagesResponse:
        return await self.async_create_variation(
           image, model, provider, response_format, **kwargs
        )
