from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Dict, Type
from ..typing import Messages, CreateResult

class BaseProvider(ABC):
    """
    Abstract base class for a provider.

    Attributes:
        url (str): URL of the provider.
        working (bool): Indicates if the provider is currently working.
        needs_auth (bool): Indicates if the provider needs authentication.
        supports_stream (bool): Indicates if the provider supports streaming.
        supports_message_history (bool): Indicates if the provider supports message history.
        supports_system_message (bool): Indicates if the provider supports system messages.
        params (str): List parameters for the provider.
    """

    url: str = None
    working: bool = False
    active_by_default: bool = None
    needs_auth: bool = False
    supports_stream: bool = False
    supports_message_history: bool = False
    supports_system_message: bool = False
    params: str
    create_function: callable
    async_create_function: callable

    @classmethod
    def get_dict(cls) -> Dict[str, str]:
        """
        Get a dictionary representation of the provider.

        Returns:
            Dict[str, str]: A dictionary with provider's details.
        """
        return {'name': cls.__name__, 'url': cls.url, 'label': getattr(cls, 'label', None)} 

    @classmethod
    def get_parent(cls) -> str:
        return getattr(cls, "parent", cls.__name__)

    @abstractmethod
    def create_function(
        *args,
        **kwargs
    ) -> CreateResult:
        """
        Create a function to generate a response based on the model and messages.

        Args:
            model (str): The model to use.
            messages (Messages): The messages to process.
            stream (bool): Whether to stream the response.

        Returns:
            CreateResult: The result of the creation.
        """
        raise NotImplementedError()

    @staticmethod
    def async_create_function(
        *args,
        **kwargs
    ) -> CreateResult:
        """
        Asynchronously create a function to generate a response based on the model and messages.

        Args:
            model (str): The model to use.
            messages (Messages): The messages to process.
            stream (bool): Whether to stream the response.

        Returns:
            CreateResult: The result of the creation.
        """
        raise NotImplementedError()

class BaseRetryProvider(BaseProvider):
    """
    Base class for a provider that implements retry logic.

    Attributes:
        providers (List[Type[BaseProvider]]): List of providers to use for retries.
        shuffle (bool): Whether to shuffle the providers list.
        exceptions (Dict[str, Exception]): Dictionary of exceptions encountered.
        last_provider (Type[BaseProvider]): The last provider used.
    """

    __name__: str = "RetryProvider"
    supports_stream: bool = True
    last_provider: Type[BaseProvider] = None

ProviderType = Union[Type[BaseProvider], BaseRetryProvider]

class Streaming():
    def __init__(self, data: str) -> None:
        self.data = data

    def __str__(self) -> str:
        return self.data
