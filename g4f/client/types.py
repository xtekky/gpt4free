from __future__ import annotations

import os

from .stubs import ChatCompletion, ChatCompletionChunk
from ..providers.types import BaseProvider
from typing import Union, Iterator, AsyncIterator

Proxies = Union[dict, str]
IterResponse = Iterator[Union[ChatCompletion, ChatCompletionChunk]]
AsyncIterResponse = AsyncIterator[Union[ChatCompletion, ChatCompletionChunk]]

class Client():
    def __init__(
        self,
        api_key: str = None,
        proxies: Proxies = None,
        **kwargs
    ) -> None:
        self.api_key: str = api_key
        self.proxies = proxies 
        self.proxy: str = self.get_proxy()

    def get_proxy(self) -> Union[str, None]:
        if isinstance(self.proxies, str):
            return self.proxies
        elif self.proxies is None:
            return os.environ.get("G4F_PROXY")
        elif "all" in self.proxies:
            return self.proxies["all"]
        elif "https" in self.proxies:
            return self.proxies["https"]