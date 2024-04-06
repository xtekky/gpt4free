from ..providers.types import BaseProvider, ProviderType
from typing import Union, Iterator
ImageProvider = Union[BaseProvider, object]
Proxies = Union[dict, str]
IterResponse = Iterator[Union[ChatCompletion, ChatCompletionChunk]]
