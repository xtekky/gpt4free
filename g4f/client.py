from __future__ import annotations

import re

from .typing import Union, Generator, AsyncGenerator, Messages, ImageType
from .base_provider import BaseProvider, ProviderType
from .Provider.base_provider import AsyncGeneratorProvider
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
    idx = 1
    chunk = None
    finish_reason = "stop"
    for idx, chunk in enumerate(response):
        content += str(chunk)
        if max_tokens is not None and idx > max_tokens:
            finish_reason = "max_tokens"
            break
        first = -1
        word = None
        if stop is not None:
            for word in list(stop):
                first = content.find(word)
                if first != -1:
                    content = content[:first]
                    break
            if stream:
                if first != -1:
                    first = chunk.find(word)
                    if first != -1:
                        chunk = chunk[:first]
                    else:
                        first = 0
            yield ChatCompletionChunk([ChatCompletionDeltaChoice(ChatCompletionDelta(chunk))])
        if first != -1:
            break
    if not stream:
        if response_format is not None and "type" in response_format:
            if response_format["type"] == "json_object":
                response = read_json(response)
        yield ChatCompletion([ChatCompletionChoice(ChatCompletionMessage(response, finish_reason))])

async def aiter_response(
    response: aiter,
    stream: bool,
    response_format: dict = None,
    max_tokens: int = None,
    stop: list = None
) -> AsyncGenerator:
    content = ""
    try:
        idx = 0
        chunk = None
        async for chunk in response:
            content += str(chunk)
            if max_tokens is not None and idx > max_tokens:
                break
            first = -1
            word = None
            if stop is not None:
                for word in list(stop):
                    first = content.find(word)
                    if first != -1:
                        content = content[:first]
                        break
            if stream:
                if first != -1:
                    first = chunk.find(word)
                    if first != -1:
                        chunk = chunk[:first]
                    else:
                        first = 0
                yield ChatCompletionChunk([ChatCompletionDeltaChoice(ChatCompletionDelta(chunk))])
            if first != -1:
                break
            idx += 1
    except:
        ...
    if not stream:
        if response_format is not None and "type" in response_format:
            if response_format["type"] == "json_object":
                response = read_json(response)
        yield ChatCompletion([ChatCompletionChoice(ChatCompletionMessage(response))])

class Model():
    def __getitem__(self, item):
        return getattr(self, item)

class ChatCompletion(Model):
    def __init__(self, choices: list):
        self.choices = choices
        
class ChatCompletionChunk(Model):
    def __init__(self, choices: list):
        self.choices = choices

class ChatCompletionChoice(Model):
    def __init__(self, message: ChatCompletionMessage):
        self.message = message
        
class ChatCompletionMessage(Model):
    def __init__(self, content: str, finish_reason: str):
        self.content = content
        self.finish_reason = finish_reason
        self.index = 0
        self.logprobs = None
        
class ChatCompletionDelta(Model):
    def __init__(self, content: str):
        self.content = content
        
class ChatCompletionDeltaChoice(Model):
    def __init__(self, delta: ChatCompletionDelta):
        self.delta = delta

class Client():
    proxies: Proxies = None
    chat: Chat

    def __init__(
        self,
        provider: ProviderType = None,
        image_provider: ImageProvider = None,
        proxies: Proxies = None,
        **kwargs
    ) -> None:
        self.proxies: Proxies = proxies
        self.images = Images(self, image_provider)
        self.chat = Chat(self, provider)

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
        stop: list = None,
        **kwargs
    ) -> Union[dict, Generator]:
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = list(stop)
        model, provider = get_model_and_provider(
            model,
            self.provider if provider is None else provider,
            stream,
            **kwargs
        )
        response = provider.create_completion(model, messages, stream=stream, **kwargs)
        if isinstance(provider, type) and issubclass(provider, AsyncGeneratorProvider):
            response = iter_response(response, stream, response_format) # max_tokens, stop
        else:
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

    def get(self, name: str) -> ImageProvider:
        return getattr(self, name) if hasattr(self, name) else self.default

class ImagesResponse(Model):
    data: list[Image]

    def __init__(self, data: list) -> None:
        self.data = data
    
class Image(Model):
    url: str

    def __init__(self, url: str) -> None:
        self.url = url
    
class Images():
    def __init__(self, client: Client, provider: ImageProvider = None):
        self.client: Client = client
        self.provider: ImageProvider = provider
        self.models: ImageModels = ImageModels(client)

    def generate(self, prompt, model: str = None, **kwargs):
        provider = self.models.get(model) if model else self.provider or self.models.get(model)
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
                return ImagesResponse([Image(image)for image in list(chunk.images)])
        raise NoImageResponseError()

    def create_variation(self, image: ImageType, model: str = None, **kwargs):
        provider = self.models.get(model) if model else self.provider
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