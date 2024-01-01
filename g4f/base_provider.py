from abc import ABC, abstractmethod
from .typing import Messages, CreateResult, Union
    
class BaseProvider(ABC):
    url: str
    working: bool = False
    needs_auth: bool = False
    supports_stream: bool = False
    supports_gpt_35_turbo: bool = False
    supports_gpt_4: bool = False
    supports_message_history: bool = False
    params: str

    @classmethod
    @abstractmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        **kwargs
    ) -> CreateResult:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        **kwargs
    ) -> str:
        raise NotImplementedError()
    
    @classmethod
    def get_dict(cls):
        return {'name': cls.__name__, 'url': cls.url} 
    
class BaseRetryProvider(BaseProvider):
    __name__: str = "RetryProvider"
    supports_stream: bool = True

    def __init__(
        self,
        providers: list[type[BaseProvider]],
        shuffle: bool = True
    ) -> None:
        self.providers: list[type[BaseProvider]] = providers
        self.shuffle: bool = shuffle
        self.working: bool = True
        self.exceptions: dict[str, Exception] = {}
        self.last_provider: type[BaseProvider] = None
        
ProviderType = Union[type[BaseProvider], BaseRetryProvider]