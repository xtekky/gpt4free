from ..providers.types import BaseProvider, ProviderType, FinishReason
from typing import Union, Iterator

ImageProvider = Union[BaseProvider, object]
Proxies = Union[dict, str]
IterResponse = Iterator[Union[ChatCompletion, ChatCompletionChunk]]

class Client():
    def __init__(
        self,
        api_key: str = None,
        proxies: Proxies = None,
        provider: ProviderType = None,
        image_provider: ImageProvider = None,
        **kwargs
    ) -> None:
        self.api_key: str = api_key
        self.proxies: Proxies = proxies
        self.chat: Chat = Chat(self, provider)
        self.images: Images = Images(self, image_provider)
