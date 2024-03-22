from ..typing import Union, Iterator, Messages
from ..stubs  import ChatCompletion, ChatCompletionChunk
from ._engine import LocalProvider
from ._models import models
from ..client import iter_response, filter_none, IterResponse

class LocalClient():
    def __init__(self, **kwargs) -> None:
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