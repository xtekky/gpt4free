from __future__ import annotations

import json
import os
import time
import random
import string
import asyncio
import aiohttp
import base64
import requests
from pathlib import Path
from datetime import datetime
from typing import Union, AsyncIterator, Iterator, Awaitable, Optional, List, Dict, Any, Type

from ..image.copy_images import copy_media, get_media_dir
from ..typing import Messages, ImageType
from ..providers.types import ProviderType, BaseProvider
from ..providers.response import *
from ..errors import NoMediaResponseError, ProviderNotFoundError
from ..providers.retry_provider import IterListProvider
from ..providers.asyncio import to_sync_generator
from ..providers.any_provider import AnyProvider
from ..Provider import OpenaiAccount, PollinationsImage, ProviderUtils
from ..Provider.template import OpenaiTemplate
from ..tools.run_tools import async_iter_run_tools, iter_run_tools
from ..cookies import get_cookies_dir
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
    stop: Optional[list[str]] = None,
    provider_info: Optional[ProviderInfo] = None
) -> ChatCompletionResponseType:
    content = ""
    reasoning = []
    finish_reason = None
    tool_calls = None
    usage = None
    provider_info: ProviderInfo = None
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
            provider_info = chunk
            continue
        elif isinstance(chunk, Reasoning):
            reasoning.append(chunk)
        elif isinstance(chunk, (HiddenResponse, Exception, JsonRequest, JsonResponse)):
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
            if provider_info is not None:
                chunk.provider = provider_info.name
                chunk.model = provider_info.model
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
            **(filter_none(
                tool_calls=[ToolCallModel.model_construct(**tool_call) for tool_call in tool_calls]
            ) if tool_calls is not None else {}),
            conversation=None if conversation is None else conversation.get_dict(),
            reasoning=reasoning if reasoning else None
        )
    if provider_info is not None:
        chat_completion.provider = provider_info.name
        chat_completion.model = provider_info.model
    yield chat_completion

async def async_iter_response(
    response: AsyncIterator[Union[str, ResponseType]],
    stream: bool,
    response_format: Optional[dict] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[list[str]] = None,
    provider_info: Optional[ProviderInfo] = None
) -> AsyncChatCompletionResponseType:
    content = ""
    reasoning = []
    finish_reason = None
    completion_id = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
    idx = 0
    tool_calls = None
    usage = None
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
                provider_info = chunk
                continue
            elif isinstance(chunk, Reasoning) and not stream:
                reasoning.append(chunk)
            elif isinstance(chunk, (HiddenResponse, Exception, JsonRequest, JsonResponse)):
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
                if provider_info is not None:
                    chunk.provider = provider_info.name
                    chunk.model = provider_info.model
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
                **(filter_none(
                    tool_calls=[ToolCallModel.model_construct(**tool_call) for tool_call in tool_calls]
                ) if tool_calls else {}),
                conversation=conversation,
                reasoning=reasoning if reasoning else None
            )
        if provider_info is not None:
            chat_completion.provider = provider_info.name
            chat_completion.model = provider_info.model
        yield chat_completion
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
        if self.base_url and provider is None:
            provider = create_custom_provider(base_url=self.base_url, api_key=self.api_key)
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
        ignore_stream: Optional[bool] = False,
        raw: Optional[bool] = False,
        **kwargs
    ) -> ChatCompletion:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        resolve_media(kwargs, image, image_name)
        if hasattr(model, "name"):
            model = model.get_long_name()
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
                api_key=self.client.api_key,
                base_url=self.client.base_url
            ),
            **kwargs
        )

        provider_info = ProviderInfo(**provider.get_dict(), model=model)

        def fallback(response):
            return iter_response(response, stream, response_format, max_tokens, stop, provider_info)

        if raw:
            def raw_response(response):
                chunks = []
                started = False
                for chunk in response:
                    if isinstance(chunk, JsonResponse):
                        yield chunk
                        started = True
                    else:
                        chunks.append(chunk)
                if not started:
                    for chunk in fallback(chunks):
                        yield chunk
            if stream:
                return raw_response(response)
            return next(raw_response(response))
        if stream:
            return fallback(response)
        return next(fallback(response))

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
        ignore_stream: Optional[bool] = False,
        raw: Optional[bool] = False,
        **kwargs
    ) -> Awaitable[ChatCompletion]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        resolve_media(kwargs, image, image_name)
        if hasattr(model, "name"):
            model = model.get_long_name()
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
                api_key=self.client.api_key,
                base_url=self.client.base_url
            ),
            **kwargs
        )

        def fallback(response):
            provider_info = ProviderInfo(**provider.get_dict(), model=model)
            return async_iter_response(response, stream, response_format, max_tokens, stop, provider_info)

        if raw:
            async def raw_response(response):
                chunks = []
                started = False
                async for chunk in response:
                    if isinstance(chunk, JsonResponse):
                        yield chunk
                        started = True
                    else:
                        chunks.append(chunk)
                if not started:
                    for chunk in fallback(chunks):
                        yield chunk
            if stream:
                return raw_response(response)
            return anext(raw_response(response))
        if stream:
            return fallback(response)
        return anext(fallback(response))

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


def create_custom_provider(
    base_url: str,
    api_key: str = None,
    name: str = None,
    working: bool = True,
    default_model: str = "",
    models: List[str] = None,
    **kwargs
) -> Type[OpenaiTemplate]:
    """
    Create a custom provider class based on OpenaiTemplate.
    
    Args:
        base_url: The base URL for the API (e.g., "https://api.example.com/v1")
        api_key: Optional API key for authentication
        name: Optional name for the provider (defaults to derived from base_url)
        working: Whether the provider is working (default: True)
        default_model: Default model to use
        models: List of available models
        **kwargs: Additional attributes to set on the provider class
    
    Returns:
        A custom provider class that extends OpenaiTemplate
    """
    if name is None:
        # Derive name from base_url
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        name = parsed.netloc.replace(".", "_").replace("-", "_").title().replace("_", "")
        if not name:
            name = "CustomProvider"
    
    # Create a new class that extends OpenaiTemplate
    class_attrs = {
        "url": base_url,
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "working": working,
        "default_model": default_model,
        "models": models or [],
        **kwargs
    }
    
    CustomProvider = type(name, (OpenaiTemplate,), class_attrs)
    print(f"Created custom provider class '{name}' with base URL '{base_url}'")
    return CustomProvider


class ClientFactory:
    """
    Factory class for creating Client and AsyncClient instances with various provider configurations.
    
    Supports:
    - Named providers (e.g., "PollinationsAI", "DeepInfra")
    - Custom providers with custom API base URLs
    - Live providers (dynamically loaded providers)
    
    Example usage:
        # Create client with a named provider
        client = ClientFactory.create_client("PollinationsAI")
                
        # Create client with custom provider
        client = ClientFactory.create_client(
            base_url="https://api.example.com/v1",
            api_key="your-api-key"
        )
        
        # Create async client
        async_client = ClientFactory.create_async_client("PollinationsAI")
    """
    
    # Registry of live/custom providers
    _live_providers_url = "https://g4f.dev/dist/js/providers.json"
    _live_providers: Dict[str, Dict] = {}
    
    @classmethod
    def create_provider(
        cls,
        name: str,
        provider: Union[Type[BaseProvider], str],
        base_url: str = None,
        api_key: str = None,
        **kwargs
    ) -> Type[BaseProvider]:
        """
        Register a live/custom provider that can be used by name.
        
        Args:
            name: Name to register the provider under
            provider: Either a provider class or "custom" to create a custom provider
            base_url: Base URL for custom providers
            api_key: API key for custom providers
            **kwargs: Additional arguments for custom provider creation
            
        Returns:
            The registered provider class
        """
        if not isinstance(provider, str):
            return provider
        elif provider.startswith("custom:"):
            if provider.startswith("custom:"):
                serverId = provider[7:]
                base_url = f"https://g4f.space/custom/{serverId}"
            if not base_url:
                raise ValueError("base_url is required for custom providers")
            provider = create_custom_provider(base_url, api_key, name=name, **kwargs)
        elif provider in ProviderUtils.convert:
            provider = ProviderUtils.convert[provider]
        else:
            if not cls._live_providers:
                path = Path(get_cookies_dir()) / "models" / datetime.today().strftime('%Y-%m-%d') / f"providers.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        cls._live_providers = json.load(f)
                cls._live_providers = requests.get(cls._live_providers_url).json()
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cls._live_providers, f, indent=4)
            if provider in cls._live_providers.get("providers", {}):
                config = cls._live_providers["providers"][provider]
                if "provider" in config and config.get("provider") in ProviderUtils.convert:
                    return ProviderUtils.convert[config.get("provider")]
                return create_custom_provider(
                    base_url=config.get("baseUrl") if api_key else config.get("backupUrl", config.get("baseUrl")),
                    api_key=api_key,
                    name=provider,
                    default_model=cls._live_providers["defaultModels"].get(provider, ""),
                )
            else:
                raise ProviderNotFoundError(f"Provider '{name}' not found")
        return provider
    
    @classmethod
    def create_client(
        cls,
        provider: Union[str, Type[BaseProvider], None] = None,
        media_provider: Union[str, Type[BaseProvider], None] = None,
        base_url: str = None,
        api_key: str = None,
        proxies: Union[dict, str] = None,
        **kwargs
    ) -> "Client":
        """
        Create a synchronous Client instance.
        
        Args:
            provider: Provider name(s), class(es), or None for default
            media_provider: Provider for media/image generation
            base_url: API base URL for custom provider
            api_key: API key for authentication
            proxies: Proxy configuration
            **kwargs: Additional arguments passed to Client
            
        Returns:
            Configured Client instance
            
        Example:
            # Named provider
            client = ClientFactory.create_client("PollinationsAI")
                        
            # Custom provider
            client = ClientFactory.create_client(
                base_url="https://api.openai.com/v1",
                api_key="sk-..."
            )
        """
        return Client(
            provider=cls.create_provider(None, provider, base_url, api_key, **kwargs),
            media_provider=media_provider,
            api_key=api_key,
            base_url=base_url,
            proxies=proxies,
            **kwargs
        )
    
    @classmethod
    def create_async_client(
        cls,
        provider: Union[str, Type[BaseProvider], None] = None,
        media_provider: Union[str, Type[BaseProvider], None] = None,
        base_url: str = None,
        api_key: str = None,
        proxies: Union[dict, str] = None,
        **kwargs
    ) -> "AsyncClient":
        """
        Create an asynchronous AsyncClient instance.
        
        Args:
            provider: Provider name(s), class(es), or None for default
            media_provider: Provider for media/image generation
            base_url: API base URL for custom provider
            api_key: API key for authentication
            proxies: Proxy configuration
            **kwargs: Additional arguments passed to AsyncClient
            
        Returns:
            Configured AsyncClient instance
            
        Example:
            # Named provider
            client = ClientFactory.create_async_client("PollinationsAI")

            # Custom provider
            client = ClientFactory.create_async_client(
                base_url="https://api.openai.com/v1",
                api_key="sk-..."
            )
        """
        return AsyncClient(
            provider=cls.create_provider(provider, base_url, api_key, **kwargs),
            media_provider=media_provider,
            api_key=api_key,
            base_url=base_url,
            proxies=proxies,
            **kwargs
        )