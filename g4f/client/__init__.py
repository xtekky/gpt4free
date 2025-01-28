from __future__ import annotations

import os
import time
import random
import string
import asyncio
import aiohttp
import base64
from typing import Union, AsyncIterator, Iterator, Awaitable, Optional

from ..image import ImageResponse, copy_images
from ..typing import Messages, ImageType
from ..providers.types import ProviderType, BaseRetryProvider
from ..providers.response import ResponseType, FinishReason, BaseConversation, SynthesizeData, ToolCalls, Usage
from ..errors import NoImageResponseError
from ..providers.retry_provider import IterListProvider
from ..providers.asyncio import to_sync_generator
from ..Provider.needs_auth import BingCreateImages, OpenaiAccount
from ..tools.run_tools import async_iter_run_tools, iter_run_tools
from .stubs import ChatCompletion, ChatCompletionChunk, Image, ImagesResponse
from .image_models import ImageModels
from .types import IterResponse, ImageProvider, Client as BaseClient
from .service import get_model_and_provider, convert_to_provider
from .helper import find_stop, filter_json, filter_none, safe_aclose
from .. import debug

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
    tool_calls = None
    usage = None
    completion_id = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
    idx = 0

    if hasattr(response, '__aiter__'):
        response = to_sync_generator(response)

    for chunk in response:
        if isinstance(chunk, FinishReason):
            finish_reason = chunk.reason
            break
        elif isinstance(chunk, ToolCalls):
            tool_calls = chunk.get_list()
            continue
        elif isinstance(chunk, Usage):
            usage = chunk
            continue
        elif isinstance(chunk, BaseConversation):
            yield chunk
            continue
        elif isinstance(chunk, SynthesizeData) or not chunk:
            continue
        elif isinstance(chunk, Exception):
            continue

        chunk = str(chunk)
        content += chunk

        if max_tokens is not None and idx + 1 >= max_tokens:
            finish_reason = "length"

        first, content, chunk = find_stop(stop, content, chunk if stream else None)

        if first != -1:
            finish_reason = "stop"

        if stream:
            yield ChatCompletionChunk.model_construct(chunk, None, completion_id, int(time.time()))

        if finish_reason is not None:
            break

        idx += 1
    if usage is None:
        usage = Usage(prompt_tokens=0, completion_tokens=idx, total_tokens=idx)

    finish_reason = "stop" if finish_reason is None else finish_reason

    if stream:
        yield ChatCompletionChunk.model_construct(
            None, finish_reason, completion_id, int(time.time()),
            usage=usage.get_dict()
        )
    else:
        if response_format is not None and "type" in response_format:
            if response_format["type"] == "json_object":
                content = filter_json(content)
        yield ChatCompletion.model_construct(
            content, finish_reason, completion_id, int(time.time()),
            usage=usage.get_dict(), **filter_none(tool_calls=tool_calls)
        )

# Synchronous iter_append_model_and_provider function
def iter_append_model_and_provider(response: ChatCompletionResponseType, last_model: str, last_provider: ProviderType) -> ChatCompletionResponseType:
    if isinstance(last_provider, BaseRetryProvider):
        last_provider = last_provider.last_provider
    for chunk in response:
        if isinstance(chunk, (ChatCompletion, ChatCompletionChunk)):
            if last_provider is not None:
                chunk.model = getattr(last_provider, "last_model", last_model)
                chunk.provider = last_provider.__name__
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
    tool_calls = None
    usage = None

    try:
        async for chunk in response:
            if isinstance(chunk, FinishReason):
                finish_reason = chunk.reason
                break
            elif isinstance(chunk, BaseConversation):
                yield chunk
                continue
            elif isinstance(chunk, ToolCalls):
                tool_calls = chunk.get_list()
                continue
            elif isinstance(chunk, Usage):
                usage = chunk
                continue
            elif isinstance(chunk, SynthesizeData) or not chunk:
                continue
            elif isinstance(chunk, Exception):
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
                yield ChatCompletionChunk.model_construct(chunk, None, completion_id, int(time.time()))

            if finish_reason is not None:
                break

        finish_reason = "stop" if finish_reason is None else finish_reason

        if usage is None:
            usage = Usage(prompt_tokens=0, completion_tokens=idx, total_tokens=idx)

        if stream:
            yield ChatCompletionChunk.model_construct(
                None, finish_reason, completion_id, int(time.time()),
                usage=usage.get_dict()
            )
        else:
            if response_format is not None and "type" in response_format:
                if response_format["type"] == "json_object":
                    content = filter_json(content)
            yield ChatCompletion.model_construct(
                content, finish_reason, completion_id, int(time.time()),
                usage=usage.get_dict(), **filter_none(tool_calls=tool_calls)
            )
    finally:
        await safe_aclose(response)

async def async_iter_append_model_and_provider(
        response: AsyncChatCompletionResponseType,
        last_model: str,
        last_provider: ProviderType
    ) -> AsyncChatCompletionResponseType:
    last_provider = None
    try:
        if isinstance(last_provider, BaseRetryProvider):
            if last_provider is not None:
                last_provider = last_provider.last_provider
        async for chunk in response:
            if isinstance(chunk, (ChatCompletion, ChatCompletionChunk)):
                if last_provider is not None:
                    chunk.model = getattr(last_provider, "last_model", last_model)
                    chunk.provider = last_provider.__name__
            yield chunk
    finally:
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
        image: Optional[ImageType] = None,
        image_name: Optional[str] = None,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[list[str], str]] = None,
        api_key: Optional[str] = None,
        ignore_working: Optional[bool] = False,
        ignore_stream: Optional[bool] = False,
        **kwargs
    ) -> ChatCompletion:
        if image is not None:
            kwargs["images"] = [(image, image_name)]
        model, provider = get_model_and_provider(
            model,
            self.provider if provider is None else provider,
            stream,
            ignore_working,
            ignore_stream,
            has_images="images" in kwargs
        )
        stop = [stop] if isinstance(stop, str) else stop
        if ignore_stream:
            kwargs["ignore_stream"] = True

        response = iter_run_tools(
            provider.get_create_function(),
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

        response = iter_response(response, stream, response_format, max_tokens, stop)
        response = iter_append_model_and_provider(response, model, provider)
        if stream:
            return response
        else:
            return next(response)

    def stream(
        self,
        messages: Messages,
        model: str,
        **kwargs
    ) -> IterResponse:
        return self.create(messages, model, stream=True, **kwargs)

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
        response_format: Optional[str] = None,
        proxy: Optional[str] = None,
        **kwargs
    ) -> ImagesResponse:
        """
        Synchronous generate method that runs the async_generate method in an event loop.
        """
        return asyncio.run(self.async_generate(prompt, model, provider, response_format, proxy, **kwargs))

    async def get_provider_handler(self, model: Optional[str], provider: Optional[ImageProvider], default: ImageProvider) -> ImageProvider:
        if provider is None:
            provider_handler = self.provider
            if provider_handler is None:
                provider_handler = self.models.get(model, default)
        elif isinstance(provider, str):
            provider_handler = convert_to_provider(provider)
        else:
            provider_handler = provider
        if provider_handler is None:
            return default
        return provider_handler

    async def async_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[ProviderType] = None,
        response_format: Optional[str] = None,
        proxy: Optional[str] = None,
        **kwargs
    ) -> ImagesResponse:
        provider_handler = await self.get_provider_handler(model, provider, BingCreateImages)
        provider_name = provider_handler.__name__ if hasattr(provider_handler, "__name__") else type(provider_handler).__name__
        if proxy is None:
            proxy = self.client.proxy

        error = None
        response = None
        if isinstance(provider_handler, IterListProvider):
            for provider in provider_handler.providers:
                try:
                    response = await self._generate_image_response(provider, provider.__name__, model, prompt, **kwargs)
                    if response is not None:
                        provider_name = provider.__name__
                        break
                except Exception as e:
                    error = e
                    debug.log(f"Image provider {provider.__name__}: {e}")
        else:
            response = await self._generate_image_response(provider_handler, provider_name, model, prompt, **kwargs)

        if isinstance(response, ImageResponse):
            return await self._process_image_response(
                response,
                model,
                provider_name,
                response_format,
                proxy
            )
        if response is None:
            if error is not None:
                raise error
            raise NoImageResponseError(f"No image response from {provider_name}")
        raise NoImageResponseError(f"Unexpected response type: {type(response)}")

    async def _generate_image_response(
        self,
        provider_handler,
        provider_name,
        model: str,
        prompt: str,
        prompt_prefix: str = "Generate a image: ",
        **kwargs
    ) -> ImageResponse:
        messages = [{"role": "user", "content": f"{prompt_prefix}{prompt}"}]
        response = None
        if hasattr(provider_handler, "create_async_generator"):
            async for item in provider_handler.create_async_generator(
                model,
                messages,
                stream=True,
                prompt=prompt,
                **kwargs
            ):
                if isinstance(item, ImageResponse):
                    response = item
                    break
        elif hasattr(provider_handler, "create_completion"):
            for item in provider_handler.create_completion(
                model,
                messages,
                True,
                prompt=prompt,
                **kwargs
            ):
                if isinstance(item, ImageResponse):
                    response = item
                    break
        else:
            raise ValueError(f"Provider {provider_name} does not support image generation")
        return response

    def create_variation(
        self,
        image: ImageType,
        model: str = None,
        provider: Optional[ProviderType] = None,
        response_format: Optional[str] = None,
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
        response_format: Optional[str] = None,
        proxy: Optional[str] = None,
        **kwargs
    ) -> ImagesResponse:
        provider_handler = await self.get_provider_handler(model, provider, OpenaiAccount)
        provider_name = provider_handler.__name__ if hasattr(provider_handler, "__name__") else type(provider_handler).__name__
        if proxy is None:
            proxy = self.client.proxy
        prompt = "create a variation of this image"
        if image is not None:
            kwargs["images"] = [(image, None)]

        error = None
        response = None
        if isinstance(provider_handler, IterListProvider):
            for provider in provider_handler.providers:
                try:
                    response = await self._generate_image_response(provider, provider.__name__, model, prompt, **kwargs)
                    if response is not None:
                        provider_name = provider.__name__
                        break
                except Exception as e:
                    error = e
                    debug.log(f"Image provider {provider.__name__}: {e}")
        else:
            response = await self._generate_image_response(provider_handler, provider_name, model, prompt, **kwargs)

        if isinstance(response, ImageResponse):
            return await self._process_image_response(response, model, provider_name, response_format, proxy)
        if response is None:
            if error is not None:
                raise error
            raise NoImageResponseError(f"No image response from {provider_name}")
        raise NoImageResponseError(f"Unexpected response type: {type(response)}")

    async def _process_image_response(
        self,
        response: ImageResponse,
        model: str,
        provider: str,
        response_format: Optional[str] = None,
        proxy: str = None
    ) -> ImagesResponse:
        if response_format == "url":
            # Return original URLs without saving locally
            images = [Image.model_construct(url=image, revised_prompt=response.alt) for image in response.get_list()]
        elif response_format == "b64_json":
            # Convert URLs directly to base64 without saving
            async def get_b64_from_url(url: str) -> Image:
                async with aiohttp.ClientSession(cookies=response.get("cookies")) as session:
                    async with session.get(url, proxy=proxy) as resp:
                        if resp.status == 200:
                            image_data = await resp.read()
                            b64_data = base64.b64encode(image_data).decode()
                            return Image.model_construct(b64_json=b64_data, revised_prompt=response.alt)
            images = await asyncio.gather(*[get_b64_from_url(image) for image in response.get_list()])
        else:
            # Save locally for None (default) case
            images = await copy_images(response.get_list(), response.get("cookies"), proxy)
            images = [Image.model_construct(url=f"/images/{os.path.basename(image)}", revised_prompt=response.alt) for image in images]
        
        return ImagesResponse.model_construct(
            created=int(time.time()),
            data=images,
            model=model,
            provider=provider
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
        image: Optional[ImageType] = None,
        image_name: Optional[str] = None,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[list[str], str]] = None,
        api_key: Optional[str] = None,
        ignore_working: Optional[bool] = False,
        ignore_stream: Optional[bool] = False,
        **kwargs
    ) -> Awaitable[ChatCompletion]:
        if image is not None:
            kwargs["images"] = [(image, image_name)]
        model, provider = get_model_and_provider(
            model,
            self.provider if provider is None else provider,
            stream,
            ignore_working,
            ignore_stream,
            has_images="images" in kwargs,
        )
        stop = [stop] if isinstance(stop, str) else stop
        if ignore_stream:
            kwargs["ignore_stream"] = True
            
        response = async_iter_run_tools(
            provider,
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

        response = async_iter_response(response, stream, response_format, max_tokens, stop)
        response = async_iter_append_model_and_provider(response, model, provider)

        if stream:
            return response
        else:
            return anext(response)

    def stream(
        self,
        messages: Messages,
        model: str,
        **kwargs
    ) -> AsyncIterator[ChatCompletionChunk, BaseConversation]:
        return self.create(messages, model, stream=True, **kwargs)

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
        response_format: Optional[str] = None,
        **kwargs
    ) -> ImagesResponse:
        return await self.async_generate(prompt, model, provider, response_format, **kwargs)

    async def create_variation(
        self,
        image: ImageType,
        model: str = None,
        provider: ProviderType = None,
        response_format: Optional[str] = None,
        **kwargs
    ) -> ImagesResponse:
        return await self.async_create_variation(
           image, model, provider, response_format, **kwargs
        )
