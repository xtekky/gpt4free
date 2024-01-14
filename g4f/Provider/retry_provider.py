from __future__ import annotations

import asyncio
import random
from ..typing import CreateResult, Messages
from ..base_provider import BaseRetryProvider
from .. import debug
from ..errors import RetryProviderError, RetryNoProviderError

class RetryProvider(BaseRetryProvider):
    """
    A provider class to handle retries for creating completions with different providers.

    Attributes:
        providers (list): A list of provider instances.
        shuffle (bool): A flag indicating whether to shuffle providers before use.
        exceptions (dict): A dictionary to store exceptions encountered during retries.
        last_provider (BaseProvider): The last provider that was used.
    """

    def create_completion(
        self,
        model: str,
        messages: Messages,
        stream: bool = False,
        **kwargs
    ) -> CreateResult:
        """
        Create a completion using available providers, with an option to stream the response.

        Args:
            model (str): The model to be used for completion.
            messages (Messages): The messages to be used for generating completion.
            stream (bool, optional): Flag to indicate if the response should be streamed. Defaults to False.

        Yields:
            CreateResult: Tokens or results from the completion.

        Raises:
            Exception: Any exception encountered during the completion process.
        """
        providers = [p for p in self.providers if stream and p.supports_stream] if stream else self.providers
        if self.shuffle:
            random.shuffle(providers)

        self.exceptions = {}
        started: bool = False
        for provider in providers:
            self.last_provider = provider
            try:
                if debug.logging:
                    print(f"Using {provider.__name__} provider")
                for token in provider.create_completion(model, messages, stream, **kwargs):
                    yield token
                    started = True
                if started:
                    return
            except Exception as e:
                self.exceptions[provider.__name__] = e
                if debug.logging:
                    print(f"{provider.__name__}: {e.__class__.__name__}: {e}")
                if started:
                    raise e

        self.raise_exceptions()

    async def create_async(
        self,
        model: str,
        messages: Messages,
        **kwargs
    ) -> str:
        """
        Asynchronously create a completion using available providers.

        Args:
            model (str): The model to be used for completion.
            messages (Messages): The messages to be used for generating completion.

        Returns:
            str: The result of the asynchronous completion.

        Raises:
            Exception: Any exception encountered during the asynchronous completion process.
        """
        providers = self.providers
        if self.shuffle:
            random.shuffle(providers)

        self.exceptions = {}
        for provider in providers:
            self.last_provider = provider
            try:
                return await asyncio.wait_for(
                    provider.create_async(model, messages, **kwargs),
                    timeout=kwargs.get("timeout", 60)
                )
            except Exception as e:
                self.exceptions[provider.__name__] = e
                if debug.logging:
                    print(f"{provider.__name__}: {e.__class__.__name__}: {e}")

        self.raise_exceptions()

    def raise_exceptions(self) -> None:
        """
        Raise a combined exception if any occurred during retries.

        Raises:
            RetryProviderError: If any provider encountered an exception.
            RetryNoProviderError: If no provider is found.
        """
        if self.exceptions:
            raise RetryProviderError("RetryProvider failed:\n" + "\n".join([
                f"{p}: {exception.__class__.__name__}: {exception}" for p, exception in self.exceptions.items()
            ]))

        raise RetryNoProviderError("No provider found")