import random, string, time, re

from ..typing import Union, Iterator, Messages
from ..stubs  import ChatCompletion, ChatCompletionChunk
from .core.engine import LocalProvider
from .core.models import models

IterResponse = Iterator[Union[ChatCompletion, ChatCompletionChunk]]

def read_json(text: str) -> dict:
    match = re.search(r"```(json|)\n(?P<code>[\S\s]+?)\n```", text)
    if match:
        return match.group("code")
    return text

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
        content += str(chunk)
        if max_tokens is not None and idx + 1 >= max_tokens:
            finish_reason = "length"
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
            yield ChatCompletionChunk(chunk, None, completion_id, int(time.time()))
        if finish_reason is not None:
            break
    finish_reason = "stop" if finish_reason is None else finish_reason
    if stream:
        yield ChatCompletionChunk(None, finish_reason, completion_id, int(time.time()))
    else:
        if response_format is not None and "type" in response_format:
            if response_format["type"] == "json_object":
                content = read_json(content)
        yield ChatCompletion(content, finish_reason, completion_id, int(time.time()))

def filter_none(**kwargs):
    for key in list(kwargs.keys()):
        if kwargs[key] is None:
            del kwargs[key]
    return kwargs

class LocalClient():
    def __init__(
        self,
        **kwargs
    ) -> None:
        self.chat: Chat = Chat(self)
        
    @staticmethod
    def list_models():
        return list(models.keys())
        
class Completions():
    def __init__(self, client: LocalClient):
        self.client: LocalClient = client

    def create(
        self,
        messages: Messages,
        model: str,
        stream: bool = False,
        response_format: dict = None,
        max_tokens: int = None,
        stop: Union[list[str], str] = None,
        **kwargs
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:

        stop = [stop] if isinstance(stop, str) else stop
        response = LocalProvider.create_completion(
            model, messages, stream,            
            **filter_none(
                max_tokens=max_tokens,
                stop=stop,
            ),
            **kwargs
        )
        response = iter_response(response, stream, response_format, max_tokens, stop)
        return response if stream else next(response)
    
class Chat():
    completions: Completions

    def __init__(self, client: LocalClient):
        self.completions = Completions(client)
    