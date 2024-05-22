from __future__ import annotations

import time
import random
import string

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

def iter_response(
    response: iter[str],
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
        chunk.provider =  last_provider.get("name")
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
        self.images: Images = Images(self, image_provider)

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
        proxy: str = None,
        response_format: dict = None,
        max_tokens: int = None,
        stop: Union[list[str], str] = None,
        api_key: str = None,
        ignored  : list[str] = None,
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
            model, messages,
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

class Chat():
    completions: Completions

    def __init__(self, client: Client, provider: ProviderType = None):
        self.completions = Completions(client, provider)

def iter_image_response(response: Iterator) -> Union[ImagesResponse, None]:
    for chunk in list(response):
        if isinstance(chunk, ImageProviderResponse):
            return ImagesResponse([Image(image) for image in chunk.get_list()])

def create_image(client: Client, provider: ProviderType, prompt: str, model: str = "", **kwargs) -> Iterator:


    if isinstance(provider, type) and provider.__name__ == "You":
        kwargs["chat_mode"] = "create"
    else:
        prompt = f"create a image with: {prompt}"
    return provider.create_completion(
        model,
        [{"role": "user", "content": prompt}],
        stream=True,
        proxy=client.get_proxy(),
        **kwargs
    )

class Images():
    def __init__(self, client: Client, provider: ImageProvider = None):
        self.client: Client = client
        self.provider: ImageProvider = provider
        self.models: ImageModels = ImageModels(client)

    def generate(self, prompt, model: str = None, **kwargs) -> ImagesResponse:
        provider = self.models.get(model, self.provider)
        if isinstance(provider, type) and issubclass(provider, BaseProvider):
            response = create_image(self.client, provider, prompt, **kwargs)
        else:
            response = list(provider.create(prompt))
        image = iter_image_response(response)
        if image is None:
            raise NoImageResponseError()
        return image

    def create_variation(self, image: ImageType, model: str = None, **kwargs):
        provider = self.models.get(model, self.provider)
        result = None
        if isinstance(provider, type) and issubclass(provider, BaseProvider):
            response = provider.create_completion(
                "",
                [{"role": "user", "content": "create a image like this"}],
                True,
                image=image,
                proxy=self.client.get_proxy(),
                **kwargs
            )
            result = iter_image_response(response)
        if result is None:
            raise NoImageResponseError()
        return result