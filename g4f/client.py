from __future__ import annotations

import re

from .stubs import ChatCompletion, ChatCompletionChunk, Image, ImagesResponse
from .typing import Union, Generator, Messages, ImageType
from .base_provider import BaseProvider, ProviderType
from .image import ImageResponse as ImageProviderResponse
from .Provider import BingCreateImages, Gemini, OpenaiChat
from .errors import NoImageResponseError
from . import get_model_and_provider

ImageProvider = Union[BaseProvider, object]
Proxies = Union[dict, str]

def read_json(text: str) -> dict:
    """
    Parses JSON code block from a string.

    Args:
        text (str): A string containing a JSON code block.

    Returns:
        dict: A dictionary parsed from the JSON code block.
    """
    match = re.search(r"```(json|)\n(?P<code>[\S\s]+?)\n```", text)
    if match:
        return match.group("code")
    return text

def iter_response(
    response: iter,
    stream: bool,
    response_format: dict = None,
    max_tokens: int = None,
    stop: list = None
) -> Generator:
    content = ""
    finish_reason = None
    last_chunk = None
    for idx, chunk in enumerate(response):
        if last_chunk is not None:
            yield ChatCompletionChunk(last_chunk, finish_reason)
        content += str(chunk)
        if max_tokens is not None and idx + 1 >= max_tokens:
            finish_reason = "max_tokens"
        first = -1
        word = None
        if stop is not None:
            for word in list(stop):
                first = content.find(word)
                if first != -1:
                    content = content[:first]
                    break
            if stream and first != -1:
                first = chunk.find(word)
                if first != -1:
                    chunk = chunk[:first]
                else:
                    first = 0
        if first != -1:
            finish_reason = "stop"
        if stream:
            last_chunk = chunk
        if finish_reason is not None:
            break
    if last_chunk is not None:
        yield ChatCompletionChunk(last_chunk, finish_reason)
    if not stream:
        if response_format is not None and "type" in response_format:
            if response_format["type"] == "json_object":
                response = read_json(response)
        yield ChatCompletion(content, finish_reason)

class Client():
    proxies: Proxies = None
    chat: Chat
    images: Images

    def __init__(
        self,
        provider: ProviderType = None,
        image_provider: ImageProvider = None,
        proxies: Proxies = None,
        **kwargs
    ) -> None:
        self.chat = Chat(self, provider)
        self.images = Images(self, image_provider)
        self.proxies: Proxies = proxies

    def get_proxy(self) -> Union[str, None]:
        if isinstance(self.proxies, str) or self.proxies is None:
            return self.proxies
        elif "all" in self.proxies:
            return self.proxies["all"]
        elif "https" in self.proxies:
            return self.proxies["https"]
        return None

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
        max_tokens: int = None,
        stop: Union[list. str] = None,
        **kwargs
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk]]:
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop
        model, provider = get_model_and_provider(
            model,
            self.provider if provider is None else provider,
            stream,
            **kwargs
        )
        response = provider.create_completion(model, messages, stream=stream, **kwargs)
        stop = [stop] if isinstance(stop, str) else stop
        response = iter_response(response, stream, response_format, max_tokens, stop)
        return response if stream else next(response)

class Chat():
    completions: Completions

    def __init__(self, client: Client, provider: ProviderType = None):
        self.completions = Completions(client, provider)

class ImageModels():
    gemini = Gemini
    openai = OpenaiChat

    def __init__(self, client: Client) -> None:
        self.client = client
        self.default = BingCreateImages(proxy=self.client.get_proxy())

    def get(self, name: str, default: ImageProvider = None) -> ImageProvider:
        return getattr(self, name) if hasattr(self, name) else default or self.default

class Images():
    def __init__(self, client: Client, provider: ImageProvider = None):
        self.client: Client = client
        self.provider: ImageProvider = provider
        self.models: ImageModels = ImageModels(client)

    def generate(self, prompt, model: str = None, **kwargs):
        provider = self.models.get(model, self.provider)
        if isinstance(provider, BaseProvider) or isinstance(provider, type) and issubclass(provider, BaseProvider):
            prompt = f"create a image: {prompt}"
            response = provider.create_completion(
                "",
                [{"role": "user", "content": prompt}],
                True,
                proxy=self.client.get_proxy(),
                **kwargs
            )
        else:
            response = provider.create(prompt)

        for chunk in response:
            if isinstance(chunk, ImageProviderResponse):
                images = [chunk.images] if isinstance(chunk.images, str) else chunk.images
                return ImagesResponse([Image(image) for image in images])
        raise NoImageResponseError()

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
            for chunk in response:
                if isinstance(chunk, ImageProviderResponse):
                    result = ([chunk.images] if isinstance(chunk.images, str) else chunk.images)
                    result = ImagesResponse([Image(image)for image in result])
        if result is None:
            raise NoImageResponseError()
        return result