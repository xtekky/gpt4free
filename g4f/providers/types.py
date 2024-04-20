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
        supports_gpt_35_turbo (bool): Indicates if the provider supports GPT-3.5 Turbo.
        supports_gpt_4 (bool): Indicates if the provider supports GPT-4.
        supports_message_history (bool): Indicates if the provider supports message history.
        params (str): List parameters for the provider.
    """

    url: str = None
    working: bool = False
    needs_auth: bool = False
    supports_stream: bool = False
    supports_gpt_35_turbo: bool = False
    supports_gpt_4: bool = False
    supports_message_history: bool = False
    supports_system_message: bool = False
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
        """
        Create a completion with the given parameters.

        Args:
            model (str): The model to use.
            messages (Messages): The messages to process.
            stream (bool): Whether to use streaming.
            **kwargs: Additional keyword arguments.

        Returns:
            CreateResult: The result of the creation process.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        **kwargs
    ) -> str:
        """
        Asynchronously create a completion with the given parameters.

        Args:
            model (str): The model to use.
            messages (Messages): The messages to process.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The result of the creation process.
        """
        raise NotImplementedError()
    
    @classmethod
    def get_dict(cls) -> Dict[str, str]:
        """
        Get a dictionary representation of the provider.

        Returns:
            Dict[str, str]: A dictionary with provider's details.
        """
        return {'name': cls.__name__, 'url': cls.url} 

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

class FinishReason():
    def __init__(self, reason: str):
        self.reason = reason

class Streaming():
    def __init__(self, data: str) -> None:
        self.data = data

    def __str__(self) -> str:
        return self.data