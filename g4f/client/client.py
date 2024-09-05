from __future__ import annotations

import os
import time
import random
import string
import logging
import asyncio
from typing import Union
from ..providers.base_provider import AsyncGeneratorProvider
from ..image import ImageResponse, to_image, to_data_uri
from ..typing import Union, Iterator, Messages, ImageType
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


def iter_response(
    response: Iterator[str],
    stream: bool,
    response_format: dict = None,
    max_tokens: int = None,
    stop: list = None
) -> IterResponse:
    content = ""
    finish_reason = None
    completion_id = ''.join(random.choices(string.ascii_letters + string.digits, k=28))
    
    for idx, chunk in enumerate(response):
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
    
    finish_reason = "stop" if finish_reason is None else finish_reason
    
    if stream:
        yield ChatCompletionChunk(None, finish_reason, completion_id, int(time.time()))
    else:
        if response_format is not None and "type" in response_format:
            if response_format["type"] == "json_object":
                content = filter_json(content)
        yield ChatCompletion(content, finish_reason, completion_id, int(time.time()))


def iter_append_model_and_provider(response: IterResponse) -> IterResponse:
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
        
        response = iter_response(response, stream, response_format, max_tokens, stop)
        response = iter_append_model_and_provider(response)
        
        return response if stream else next(response)


class Chat:
    completions: Completions

    def __init__(self, client: Client, provider: ProviderType = None):
        self.completions = Completions(client, provider)


def iter_image_response(response: Iterator) -> Union[ImagesResponse, None]:
    logging.info("Starting iter_image_response")
    response_list = list(response)
    logging.info(f"Response list: {response_list}")
    
    for chunk in response_list:
        logging.info(f"Processing chunk: {chunk}")
        if isinstance(chunk, ImageProviderResponse):
            logging.info("Found ImageProviderResponse")
            return ImagesResponse([Image(image) for image in chunk.get_list()])
    
    logging.warning("No ImageProviderResponse found in the response")
    return None


def create_image(client: Client, provider: ProviderType, prompt: str, model: str = "", **kwargs) -> Iterator:
    logging.info(f"Creating image with provider: {provider}, model: {model}, prompt: {prompt}")
    
    if isinstance(provider, type) and provider.__name__ == "You":
        kwargs["chat_mode"] = "create"
    else:
        prompt = f"create an image with: {prompt}"
    
    response = provider.create_completion(
        model,
        [{"role": "user", "content": prompt}],
        stream=True,
        proxy=client.get_proxy(),
        **kwargs
    )
    
    logging.info(f"Response from create_completion: {response}")
    return response


class Images:
    def __init__(self, client: 'Client', provider: ImageProvider = None):
        self.client: 'Client' = client
        self.provider: ImageProvider = provider
        self.models: ImageModels = ImageModels(client)

    def generate(self, prompt: str, model: str = None, **kwargs) -> ImagesResponse:
        logging.info(f"Starting synchronous image generation for model: {model}, prompt: {prompt}")
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(self.async_generate(prompt, model, **kwargs))
            logging.info(f"Synchronous image generation completed. Result: {result}")
            return result
        except Exception as e:
            logging.error(f"Error in synchronous image generation: {str(e)}")
            raise
        finally:
            if loop.is_running():
                loop.close()

    async def async_generate(self, prompt: str, model: str = None, **kwargs) -> ImagesResponse:
        logging.info(f"Generating image for model: {model}, prompt: {prompt}")
        provider = self.models.get(model, self.provider)
        if provider is None:
            raise ValueError(f"Unknown model: {model}")
        
        logging.info(f"Provider: {provider}")
        
        if isinstance(provider, IterListProvider):
            if provider.providers:
                provider = provider.providers[0]
                logging.info(f"Using first provider from IterListProvider: {provider}")
            else:
                raise ValueError(f"IterListProvider for model {model} has no providers")

        if isinstance(provider, type) and issubclass(provider, AsyncGeneratorProvider):
            logging.info("Using AsyncGeneratorProvider")
            messages = [{"role": "user", "content": prompt}]
            async for response in provider.create_async_generator(model, messages, **kwargs):
                if isinstance(response, ImageResponse):
                    return self._process_image_response(response)
                elif isinstance(response, str):
                    image_response = ImageResponse([response], prompt)
                    return self._process_image_response(image_response)
        elif hasattr(provider, 'create'):
            logging.info("Using provider's create method")
            if asyncio.iscoroutinefunction(provider.create):
                response = await provider.create(prompt)
            else:
                response = provider.create(prompt)
            
            if isinstance(response, ImageResponse):
                return self._process_image_response(response)
            elif isinstance(response, str):
                image_response = ImageResponse([response], prompt)
                return self._process_image_response(image_response)
        else:
            raise ValueError(f"Provider {provider} does not support image generation")
        
        logging.error(f"Unexpected response type: {type(response)}")
        raise NoImageResponseError(f"Unexpected response type: {type(response)}")

    def _process_image_response(self, response: ImageResponse) -> ImagesResponse:
        processed_images = []
        for image_data in response.get_list():
            if image_data.startswith('http://') or image_data.startswith('https://'):
                processed_images.append(Image(url=image_data))
            else:
                image = to_image(image_data)
                file_name = self._save_image(image)
                processed_images.append(Image(url=file_name))
        return ImagesResponse(processed_images)

    def _save_image(self, image: 'PILImage') -> str:
        os.makedirs('generated_images', exist_ok=True)
        file_name = f"generated_images/image_{int(time.time())}.png"
        image.save(file_name)
        return file_name

    async def create_variation(self, image: Union[str, bytes], model: str = None, **kwargs):
        provider = self.models.get(model, self.provider)
        if provider is None:
            raise ValueError(f"Unknown model: {model}")

        if isinstance(provider, type) and issubclass(provider, AsyncGeneratorProvider):
            messages = [{"role": "user", "content": "create a variation of this image"}]
            image_data = to_data_uri(image)
            async for response in provider.create_async_generator(model, messages, image=image_data, **kwargs):
                if isinstance(response, ImageResponse):
                    return self._process_image_response(response)
                elif isinstance(response, str):
                    image_response = ImageResponse([response], "Image variation")
                    return self._process_image_response(image_response)
        elif hasattr(provider, 'create_variation'):
            if asyncio.iscoroutinefunction(provider.create_variation):
                response = await provider.create_variation(image, **kwargs)
            else:
                response = provider.create_variation(image, **kwargs)
            
            if isinstance(response, ImageResponse):
                return self._process_image_response(response)
            elif isinstance(response, str):
                image_response = ImageResponse([response], "Image variation")
                return self._process_image_response(image_response)
        else:
            raise ValueError(f"Provider {provider} does not support image variation")

        raise NoImageResponseError("Failed to create image variation")

