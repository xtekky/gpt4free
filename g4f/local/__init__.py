from __future__ import annotations

from ..typing import Union, Messages
from ..locals.provider import LocalProvider
from ..locals.models import get_models
from ..client.client import iter_response, filter_none
from ..client.types import IterResponse

class LocalClient():
    def __init__(self, **kwargs) -> None:
        self.chat: Chat = Chat(self)

    @staticmethod
    def list_models():
        return list(get_models())

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
    ) -> IterResponse:
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