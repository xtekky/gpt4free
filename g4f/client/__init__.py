from __future__ import annotations

import os
import time
import random
import string
import asyncio
import aiohttp
import base64
from typing import Union, AsyncIterator, Iterator, Awaitable, Optional

from ..image.copy_images import copy_media, get_media_dir
from ..typing import Messages, ImageType
from ..providers.types import ProviderType, BaseRetryProvider
from ..providers.response import *
from ..errors import NoMediaResponseError
from ..providers.retry_provider import IterListProvider
from ..providers.asyncio import to_sync_generator
from ..providers.any_provider import AnyProvider
from ..Provider import OpenaiAccount, PollinationsImage
from ..tools.run_tools import async_iter_run_tools, iter_run_tools
from .stubs import ChatCompletion, ChatCompletionChunk, Image, ImagesResponse, UsageModel, ToolCallModel, ClientResponse
from .models import ClientModels
from .types import IterResponse, Client as BaseClient
from .service import convert_to_provider
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

def add_chunk(content, chunk):
    if content == "" and isinstance(chunk, (MediaResponse, AudioResponse)):
        content = chunk
    elif not isinstance(chunk, (Reasoning, ToolCalls)):
        content = str(content) + str(chunk)
    return content

def resolve_media(kwargs: dict, image = None, image_name: str = None) -> None:
    if image is not None:
        kwargs["media"] = [(image, getattr(image, "name", image_name))]
    elif "images" in kwargs:
        kwargs["media"] = kwargs.pop("images")
    if kwargs.get("media") is None:
        kwargs.pop("media", None)
    elif not isinstance(kwargs["media"], list):
        kwargs["media"] = [kwargs["media"]]
    for idx, media in enumerate(kwargs.get("media", [])):
        if not isinstance(media, (list, tuple)):
            kwargs["media"][idx] = (media, getattr(media, "name", None))

# Synchronous iter_response function
def iter_response(
    response: Union[Iterator[Union[str, ResponseType]]],
    stream: bool,
    response_format: Optional[dict] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[list[str]] = None
) -> ChatCompletionResponseType:
    content = ""
    reasoning_content = []
    finish_reason = None
    tool_calls = None
    usage = None
    provider: ProviderInfo = None
    conversation: JsonConversation = None
    completion_id = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
    idx = 0

    if hasattr(response, '__aiter__'):
        response = to_sync_generator(response)

    for chunk in response:
        if isinstance(chunk, FinishReason):
            finish_reason = chunk.reason
            continue
        elif isinstance(chunk, JsonConversation):
            conversation = chunk
            continue
        elif isinstance(chunk, ToolCalls):
            if not stream:
                tool_calls = chunk.get_list()
                continue
        elif isinstance(chunk, Usage):
            usage = chunk
            continue
        elif isinstance(chunk, ProviderInfo):
            provider = chunk
            continue
        elif isinstance(chunk, Reasoning):
            reasoning_content.append(chunk)
        elif isinstance(chunk, HiddenResponse):
            continue
        elif isinstance(chunk, Exception):
            continue
        elif not chunk:
            continue

        content = add_chunk(content, chunk)
        idx += 1

        if max_tokens is not None and idx >= max_tokens:
            finish_reason = "length"

        first, content, chunk = find_stop(stop, content, chunk if stream else None)

        if first != -1:
            finish_reason = "stop"

        if stream:
            chunk = ChatCompletionChunk.model_construct(chunk, None, completion_id, int(time.time()))
            if provider is not None:
                chunk.provider = provider.name
                chunk.model = provider.model
            yield chunk

        if finish_reason is not None:
            break

    if usage is None:
        usage = UsageModel.model_construct(completion_tokens=idx, total_tokens=idx)
    else:
        usage = UsageModel.model_construct(**usage.get_dict())

    finish_reason = "stop" if finish_reason is None else finish_reason

    if stream:
        chat_completion = ChatCompletionChunk.model_construct(
            None, finish_reason, completion_id, int(time.time()), usage=usage
        )
    else:
        if response_format is not None and "type" in response_format:
            if response_format["type"] == "json_object":
                content = filter_json(content)
        chat_completion = ChatCompletion.model_construct(
            content, finish_reason, completion_id, int(time.time()), usage=usage,
            **filter_none(tool_calls=[ToolCallModel.model_construct(**tool_call) for tool_call in tool_calls]) if tool_calls is not None else {},
            conversation=None if conversation is None else conversation.get_dict(),
            reasoning_content=reasoning_content if reasoning_content else None
        )
    if provider is not None:
        chat_completion.provider = provider.name
        chat_completion.model = provider.model
    yield chat_completion

# Synchronous iter_append_model_and_provider function
def iter_append_model_and_provider(response: ChatCompletionResponseType, last_model: str, last_provider: ProviderType) -> ChatCompletionResponseType:
    if isinstance(last_provider, BaseRetryProvider):
        yield from response
        return
    for chunk in response:
        if isinstance(chunk, (ChatCompletion, ChatCompletionChunk)):
            if chunk.provider is None and last_provider is not None:
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
    reasoning_content = []
    finish_reason = None
    completion_id = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
    idx = 0
    tool_calls = None
    usage = None
    provider: ProviderInfo = None
    conversation: JsonConversation = None

    try:
        async for chunk in response:
            if isinstance(chunk, FinishReason):
                finish_reason = chunk.reason
                continue
            elif isinstance(chunk, JsonConversation):
                conversation = chunk
                continue
            elif isinstance(chunk, ToolCalls):
                if not stream:
                    tool_calls = chunk.get_list()
                    continue
            elif isinstance(chunk, Usage):
                usage = chunk
                continue
            elif isinstance(chunk, ProviderInfo):
                provider = chunk
                continue
            elif isinstance(chunk, Reasoning) and not stream:
                reasoning_content.append(chunk)
            elif isinstance(chunk, HiddenResponse):
                continue
            elif isinstance(chunk, Exception):
                continue
            elif not chunk:
                continue

            content = add_chunk(content, chunk)
            idx += 1

            if max_tokens is not None and idx >= max_tokens:
                finish_reason = "length"

            first, content, chunk = find_stop(stop, content, chunk if stream else None)

            if first != -1:
                finish_reason = "stop"

            if stream:
                chunk = ChatCompletionChunk.model_construct(chunk, None, completion_id, int(time.time()))
                if provider is not None:
                    chunk.provider = provider.name
                    chunk.model = provider.model
                yield chunk

            if finish_reason is not None:
                break

        finish_reason = "stop" if finish_reason is None else finish_reason

        if usage is None:
            usage = UsageModel.model_construct(completion_tokens=idx, total_tokens=idx)
        else:
            usage = UsageModel.model_construct(**usage.get_dict())

        if stream:
            chat_completion = ChatCompletionChunk.model_construct(
                None, finish_reason, completion_id, int(time.time()), usage=usage, conversation=conversation
            )
        else:
            if response_format is not None and "type" in response_format:
                if response_format["type"] == "json_object":
                    content = filter_json(content)
            chat_completion = ChatCompletion.model_construct(
                content, finish_reason, completion_id, int(time.time()), usage=usage,
                **filter_none(
                    tool_calls=[ToolCallModel.model_construct(**tool_call) for tool_call in tool_calls]
                ) if tool_calls is not None else {},
                conversation=conversation,
                reasoning_content=reasoning_content if reasoning_content else None
            )
        if provider is not None:
            chat_completion.provider = provider.name
            chat_completion.model = provider.model
        yield chat_completion
    finally:
        await safe_aclose(response)

async def async_response(
    response: AsyncIterator[Union[str, ResponseType]]
) -> ClientResponse:
    content = ""
    response_id = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
    idx = 0
    usage = None
    provider: ProviderInfo = None
    conversation: JsonConversation = None

    async for chunk in response:
        if isinstance(chunk, FinishReason):
            continue
        elif isinstance(chunk, JsonConversation):
            conversation = chunk
            continue
        elif isinstance(chunk, ToolCalls):
            continue
        elif isinstance(chunk, Usage):
            usage = chunk
            continue
        elif isinstance(chunk, ProviderInfo):
            provider = chunk
            continue
        elif isinstance(chunk, HiddenResponse):
            continue
        elif isinstance(chunk, Exception):
            continue

        content = add_chunk(content, chunk)
        if not content:
            continue
        idx += 1

    if usage is None:
        usage = UsageModel.model_construct(completion_tokens=idx, total_tokens=idx)
    else:
        usage = UsageModel.model_construct(**usage.get_dict())

    response = ClientResponse.model_construct(
        content, response_id, int(time.time()), usage=usage, conversation=conversation
    )
    if provider is not None:
        response.provider = provider.name
        response.model = provider.model
    return response

async def async_iter_append_model_and_provider(
        response: AsyncChatCompletionResponseType,
        last_model: str,
        last_provider: ProviderType
    ) -> AsyncChatCompletionResponseType:
    try:
        if isinstance(last_provider, BaseRetryProvider):
            async for chunk in response:
                yield chunk
            return
        async for chunk in response:
            if isinstance(chunk, (ChatCompletion, ChatCompletionChunk)):
                if chunk.provider is None and last_provider is not None:
                    chunk.model = getattr(last_provider, "last_model", last_model)
                    chunk.provider = last_provider.__name__
            yield chunk
    finally:
        await safe_aclose(response)

class Client(BaseClient):
    def __init__(
        self,
        provider: Optional[ProviderType] = None,
        media_provider: Optional[ProviderType] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.chat: Chat = Chat(self, provider)
        if media_provider is None:
            media_provider = kwargs.get("image_provider", provider)
        self.models: ClientModels = ClientModels(self, provider, media_provider)
        self.images: Images = Images(self, media_provider)
        self.media: Images = self.images

class Completions:
    def __init__(self, client: Client, provider: Optional[ProviderType] = None):
        self.client: Client = client
        self.provider: ProviderType = provider

    def create(
        self,
        messages: Messages,
        model: str = "",
        provider: Optional[ProviderType] = None,
        stream: Optional[bool] = False,
        proxy: Optional[str] = None,
        image: Optional[ImageType] = None,
        image_name: Optional[str] = None,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[list[str], str]] = None,
        api_key: Optional[str] = None,
        ignore_stream: Optional[bool] = False,
        **kwargs
    ) -> ChatCompletion:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        resolve_media(kwargs, image, image_name)
        if hasattr(model, "name"):
            model = model.name
        if provider is None:
            provider = self.provider
            if provider is None:
                provider = AnyProvider
        if isinstance(provider, str):
            provider = convert_to_provider(provider)
        stop = [stop] if isinstance(stop, str) else stop
        if ignore_stream:
            kwargs["ignore_stream"] = True

        response = iter_run_tools(
            provider,
            model=model,
            messages=messages,
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
        model: str = "",
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

    async def get_provider_handler(self, model: Optional[str], provider: Optional[ProviderType], default: ProviderType) -> ProviderType:
        if provider is None:
            provider_handler = self.provider
            if provider_handler is None:
                provider_handler = self.client.models.get(model, default)
        else:
            provider_handler = provider
        if isinstance(provider_handler, str):
            provider_handler = convert_to_provider(provider_handler)
        return provider_handler

    async def async_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[ProviderType] = None,
        response_format: Optional[str] = None,
        proxy: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> ImagesResponse:
        provider_handler = await self.get_provider_handler(model, provider, PollinationsImage)
        provider_name = provider_handler.__name__ if hasattr(provider_handler, "__name__") else type(provider_handler).__name__
        if proxy is None:
            proxy = self.client.proxy
        if api_key is None:
            api_key = self.client.api_key
        error = None
        response = None
        if isinstance(provider_handler, IterListProvider):
            for provider in provider_handler.providers:
                try:
                    response = await self._generate_image_response(provider, provider.__name__, model, prompt, proxy=proxy, api_key=api_key, **kwargs)
                    if response is not None:
                        provider_name = provider.__name__
                        break
                except Exception as e:
                    error = e
                    debug.error(f"{provider.__name__} {type(e).__name__}: {e}")
        else:
            response = await self._generate_image_response(provider_handler, provider_name, model, prompt, proxy=proxy, api_key=api_key, **kwargs)
        if response is None:
            if error is not None:
                raise error
            raise NoMediaResponseError(f"No media response from {provider_name}")
        return await self._process_image_response(
            response,
            model,
            provider_name,
            kwargs.get("download_media", True),
            response_format,
            proxy
        )

    async def _generate_image_response(
        self,
        provider_handler: ProviderType,
        provider_name: str,
        model: str,
        prompt: str,
        prompt_prefix: str = "Generate a image: ",
        api_key: str = None,
        **kwargs
    ) -> MediaResponse:
        messages = [{"role": "user", "content": f"{prompt_prefix}{prompt}"}]
        items: list[MediaResponse] = []
        if isinstance(api_key, dict):
            api_key = api_key.get(provider_handler.get_parent())
        if hasattr(provider_handler, "create_async_generator"):
            async for item in provider_handler.create_async_generator(
                model,
                messages,
                stream=True,
                prompt=prompt,
                api_key=api_key,
                **kwargs
            ):
                if isinstance(item, (MediaResponse, AudioResponse)) and not isinstance(item, HiddenResponse):
                    items.append(item)
        elif hasattr(provider_handler, "create_completion"):
            for item in provider_handler.create_completion(
                model,
                messages,
                True,
                prompt=prompt,
                api_key=api_key,
                **kwargs
            ):
                if isinstance(item, (MediaResponse, AudioResponse)) and not isinstance(item, HiddenResponse):
                    items.append(item)
        else:
            raise ValueError(f"Provider {provider_name} does not support image generation")
        urls = []
        for item in items:
            if isinstance(item, AudioResponse):
                urls.append(item.to_uri())
            elif isinstance(item.urls, str):
                urls.append(item.urls)
            elif isinstance(item.urls, list):
                urls.extend(item.urls)
        if not urls:
            return None
        alt = getattr(items[0], "alt", "")
        return MediaResponse(urls, alt, items[0].options)

    def create_variation(
        self,
        image: ImageType,
        image_name: str = None,
        prompt: str = "Create a variation of this image",
        model: str = None,
        provider: Optional[ProviderType] = None,
        response_format: Optional[str] = None,
        **kwargs
    ) -> ImagesResponse:
        return asyncio.run(self.async_create_variation(
           image=image,
           image_name=image_name,
           prompt=prompt,
           model=model,
           provider=provider,
           response_format=response_format,
           **kwargs
        ))

    async def async_create_variation(
        self,
        *,
        image: ImageType,
        image_name: str = None,
        prompt: str = "Create a variation of this image",
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
        resolve_media(kwargs, image, image_name)
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
                    debug.error(f"{provider.__name__} {type(e).__name__}: {e}")
        else:
            response = await self._generate_image_response(provider_handler, provider_name, model, prompt, **kwargs)
        if response is None:
            if error is not None:
                raise error
            raise NoMediaResponseError(f"No media response from {provider_name}")
        return await self._process_image_response(
            response,
            model,
            provider_name,
            kwargs.get("download_media", True),
            response_format,
            proxy
        )

    async def _process_image_response(
        self,
        response: MediaResponse,
        model: str,
        provider: str,
        download_media: bool,
        response_format: Optional[str] = None,
        proxy: str = None
    ) -> ImagesResponse:
        if response_format == "url":
            # Return original URLs without saving locally
            images = [Image.model_construct(url=image, revised_prompt=response.alt) for image in response.get_list()]
        elif response_format == "b64_json":
            # Convert URLs directly to base64 without saving
            async def get_b64_from_url(url: str) -> Image:
                if url.startswith("/media/"):
                    with open(os.path.join(get_media_dir(), os.path.basename(url)), "rb") as f:
                        b64_data = base64.b64encode(f.read()).decode()
                        return Image.model_construct(b64_json=b64_data, revised_prompt=response.alt)
                async with aiohttp.ClientSession(cookies=response.get("cookies")) as session:
                    async with session.get(url, proxy=proxy) as resp:
                        if resp.status == 200:
                            b64_data = base64.b64encode(await resp.read()).decode()
                            return Image.model_construct(b64_json=b64_data, revised_prompt=response.alt)
                return Image.model_construct(url=url, revised_prompt=response.alt)
            images = await asyncio.gather(*[get_b64_from_url(image) for image in response.get_list()])
        else:
            # Save locally for None (default) case
            images = response.get_list()
            if download_media or response.get("cookies") or response.get("headers"):
                images = await copy_media(response.get_list(), response.get("cookies"), response.get("headers"), proxy, response.alt)
            images = [Image.model_construct(url=image, revised_prompt=response.alt) for image in images]

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
        media_provider: Optional[ProviderType] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.chat: AsyncChat = AsyncChat(self, provider)
        if media_provider is None:
            media_provider = kwargs.get("image_provider", provider)
        self.models: ClientModels = ClientModels(self, provider, media_provider)
        self.images: AsyncImages = AsyncImages(self, media_provider)
        self.media: AsyncImages = self.images
        self.responses: AsyncResponses = AsyncResponses(self, provider)

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
        model: str = "",
        provider: Optional[ProviderType] = None,
        stream: Optional[bool] = False,
        proxy: Optional[str] = None,
        image: Optional[ImageType] = None,
        image_name: Optional[str] = None,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[list[str], str]] = None,
        api_key: Optional[str] = None,
        ignore_stream: Optional[bool] = False,
        **kwargs
    ) -> Awaitable[ChatCompletion]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        resolve_media(kwargs, image, image_name)
        if hasattr(model, "name"):
            model = model.name
        if provider is None:
            provider = self.provider
            if provider is None:
                provider = AnyProvider
        if isinstance(provider, str):
            provider = convert_to_provider(provider)
        stop = [stop] if isinstance(stop, str) else stop
        if ignore_stream:
            kwargs["ignore_stream"] = True
            
        response = async_iter_run_tools(
            provider,
            model=model,
            messages=messages,
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
        model: str = "",
        **kwargs
    ) -> AsyncIterator[ChatCompletionChunk]:
        return self.create(messages, model, stream=True, **kwargs)

class AsyncImages(Images):
    def __init__(self, client: AsyncClient, provider: Optional[ProviderType] = None):
        self.client: AsyncClient = client
        self.provider: Optional[ProviderType] = provider

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
           image=image, model=model, provider=provider, response_format=response_format, **kwargs
        )

class AsyncResponses():
    def __init__(self, client: AsyncClient, provider: Optional[ProviderType] = None):
        self.client: AsyncClient = client
        self.provider: ProviderType = provider

    async def create(
        self,
        input: str,
        model: str = "",
        provider: Optional[ProviderType] = None,
        instructions: Optional[str] = None,
        proxy: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> ClientResponse:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        if instructions is not None:
            input = [{"role": "developer", "content": instructions}] + input
        for idx, message in enumerate(input):
            if isinstance(message["content"], list):
                for key, value in enumerate(message["content"]):
                    if isinstance(value, dict) and value.get("type") == "input_text":
                        message["content"][key] = {"type": "text", "text": value.get("text")}
                    input[idx] = {"role": message["role"], "content": message["content"]}
        resolve_media(kwargs)
        if hasattr(model, "name"):
            model = model.name
        if provider is None:
            provider = self.provider
            if provider is None:
                provider = AnyProvider
        if isinstance(provider, str):
            provider = convert_to_provider(provider)

        response = async_iter_run_tools(
            provider,
            model=model,
            messages=input,
            **filter_none(
                proxy=self.client.proxy if proxy is None else proxy,
                api_key=self.client.api_key if api_key is None else api_key
            ),
            **kwargs
        )

        return await async_response(response)