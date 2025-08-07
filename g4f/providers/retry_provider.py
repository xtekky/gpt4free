from __future__ import annotations

import random

from ..typing import Dict, Type, List, CreateResult, Messages, AsyncResult
from .types import BaseProvider, BaseRetryProvider, ProviderType
from .response import ProviderInfo, JsonConversation, is_content
from .. import debug
from ..tools.run_tools import AuthManager
from ..errors import RetryProviderError, RetryNoProviderError, MissingAuthError, NoValidHarFileError

class RotatedProvider(BaseRetryProvider):
    """
    A provider that rotates through a list of providers, attempting one provider per
    request and advancing to the next one upon failure. This distributes load and
    retries across multiple providers in a round-robin fashion.
    """
    def __init__(
        self,
        providers: List[Type[BaseProvider]],
        shuffle: bool = True
    ) -> None:
        """
        Initialize the RotatedProvider.
        Args:
            providers (List[Type[BaseProvider]]): A non-empty list of providers to rotate through.
            shuffle (bool): If True, shuffles the provider list once at initialization
                            to randomize the rotation order.
        """
        if not isinstance(providers, list) or len(providers) == 0:
            raise ValueError('RotatedProvider requires a non-empty list of providers.')
        
        self.providers = providers
        if shuffle:
            random.shuffle(self.providers)
            
        self.current_index = 0
        self.last_provider: Type[BaseProvider] = None

    def _get_current_provider(self) -> Type[BaseProvider]:
        """Gets the provider at the current index."""
        return self.providers[self.current_index]

    def _rotate_provider(self) -> None:
        """Rotates to the next provider in the list."""
        self.current_index = (self.current_index + 1) % len(self.providers)
        #new_provider_name = self.providers[self.current_index].__name__
        #debug.log(f"Rotated to next provider: {new_provider_name}")

    def create_completion(
        self,
        model: str,
        messages: Messages,
        ignored: list[str] = [], # 'ignored' is less relevant now but kept for compatibility
        api_key: str = None,
        **kwargs,
    ) -> CreateResult:
        """
        Create a completion using the current provider and rotating on failure.
        
        It will try each provider in the list once per call, rotating after each
        failed attempt, until one succeeds or all have failed.
        """
        exceptions: Dict[str, Exception] = {}
        
        # Loop over the number of providers, giving each one a chance
        for _ in range(len(self.providers)):
            provider = self._get_current_provider()
            self.last_provider = provider
            self._rotate_provider()

            # Skip if provider is in the ignored list
            if provider.get_parent() in ignored:
                continue
            
            alias = model or getattr(provider, "default_model", None)
            if hasattr(provider, "model_aliases"):
                alias = provider.model_aliases.get(model, model)
            if isinstance(alias, list):
                alias = random.choice(alias)
            
            debug.log(f"Attempting provider: {provider.__name__} with model: {alias}")
            yield ProviderInfo(**provider.get_dict(), model=alias, alias=model)
            
            extra_body = kwargs.copy()
            current_api_key = api_key.get(provider.get_parent()) if isinstance(api_key, dict) else api_key
            if not current_api_key:
                current_api_key = AuthManager.load_api_key(provider)
            if current_api_key:
                extra_body["api_key"] = current_api_key
            
            try:
                # Attempt to get a response from the current provider
                response = provider.create_function(alias, messages, **extra_body)
                started = False
                for chunk in response:
                    if chunk:
                        yield chunk
                        if is_content(chunk):
                            started = True
                if started:
                    # Success, so we return and do not rotate
                    return
            except Exception as e:
                exceptions[provider.__name__] = e
                debug.error(f"{provider.__name__} failed: {e}")
        
        # If the loop completes, all providers have failed
        raise_exceptions(exceptions)

    async def create_async_generator(
        self,
        model: str,
        messages: Messages,
        ignored: list[str] = [],
        api_key: str = None,
        conversation: JsonConversation = None,
        **kwargs
    ) -> AsyncResult:
        """
        Asynchronously create a completion, rotating through providers on failure.
        """
        exceptions: Dict[str, Exception] = {}

        for _ in range(len(self.providers)):
            provider = self._get_current_provider()
            self._rotate_provider()
            self.last_provider = provider

            if provider.get_parent() in ignored:
                continue

            alias = model or getattr(provider, "default_model", None)
            if hasattr(provider, "model_aliases"):
                alias = provider.model_aliases.get(model, model)
            if isinstance(alias, list):
                alias = random.choice(alias)
            
            debug.log(f"Attempting provider: {provider.__name__} with model: {alias}")
            yield ProviderInfo(**provider.get_dict(), model=alias)
            
            extra_body = kwargs.copy()
            current_api_key = api_key.get(provider.get_parent()) if isinstance(api_key, dict) else api_key
            if not current_api_key:
                current_api_key = AuthManager.load_api_key(provider)
            if current_api_key:
                extra_body["api_key"] = current_api_key
            if conversation and hasattr(conversation, provider.__name__):
                extra_body["conversation"] = JsonConversation(**getattr(conversation, provider.__name__))
            
            try:
                response = provider.async_create_function(alias, messages, **extra_body)
                started = False
                async for chunk in response:
                    if isinstance(chunk, JsonConversation):
                        if conversation is None: conversation = JsonConversation()
                        setattr(conversation, provider.__name__, chunk.get_dict())
                        yield conversation
                    elif chunk:
                        yield chunk
                        if is_content(chunk):
                            started = True
                if started:
                    return # Success
            except Exception as e:
                exceptions[provider.__name__] = e
                debug.error(f"{provider.__name__} failed: {e}")
                
        raise_exceptions(exceptions)

    # Maintain API compatibility
    create_function = create_completion
    async_create_function = create_async_generator

class IterListProvider(BaseRetryProvider):
    def __init__(
        self,
        providers: List[Type[BaseProvider]] = [],
        shuffle: bool = True
    ) -> None:
        """
        Initialize the BaseRetryProvider.
        Args:
            providers (List[Type[BaseProvider]]): List of providers to use.
            shuffle (bool): Whether to shuffle the providers list.
            single_provider_retry (bool): Whether to retry a single provider if it fails.
            max_retries (int): Maximum number of retries for a single provider.
        """
        self.providers = providers
        self.shuffle = shuffle
        self.working = True
        self.last_provider: Type[BaseProvider] = None

    def create_completion(
        self,
        model: str,
        messages: Messages,
        ignored: list[str] = [],
        api_key: str = None,
        **kwargs,
    ) -> CreateResult:
        """
        Create a completion using available providers.
        Args:
            model (str): The model to be used for completion.
            messages (Messages): The messages to be used for generating completion.
        Yields:
            CreateResult: Tokens or results from the completion.
        Raises:
            Exception: Any exception encountered during the completion process.
        """
        exceptions = {}
        started: bool = False
        for provider in self.get_providers(ignored):
            self.last_provider = provider
            alias = model
            if not model:
                alias = getattr(provider, "default_model", None)
            if hasattr(provider, "model_aliases"):
                alias = provider.model_aliases.get(model, model)
            if isinstance(alias, list):
                alias = random.choice(alias)
            debug.log(f"Using provider: {provider.__name__} with model: {alias}")
            yield ProviderInfo(**provider.get_dict(), model=alias)
            extra_body = kwargs.copy()
            if isinstance(api_key, dict):
                api_key = api_key.get(provider.get_parent())
            if not api_key:
                api_key = AuthManager.load_api_key(provider)
            if api_key:
                extra_body["api_key"] = api_key
            try:
                response = provider.create_function(alias, messages, **extra_body)
                for chunk in response:
                    if chunk:
                        yield chunk
                        if is_content(chunk):
                            started = True
                if started:
                    return
            except Exception as e:
                exceptions[provider.__name__] = e
                debug.error(f"{provider.__name__}:", e)
                if started:
                    raise e
                yield e

        raise_exceptions(exceptions)

    async def create_async_generator(
        self,
        model: str,
        messages: Messages,
        ignored: list[str] = [],
        api_key: str = None,
        conversation: JsonConversation = None,
        **kwargs
    ) -> AsyncResult:
        exceptions = {}
        started: bool = False

        for provider in self.get_providers(ignored):
            self.last_provider = provider
            alias = model
            if not model:
                alias = getattr(provider, "default_model", None)
            if hasattr(provider, "model_aliases"):
                alias = provider.model_aliases.get(model, model)
            if isinstance(alias, list):
                alias = random.choice(alias)
            debug.log(f"Using {provider.__name__} provider with model {alias}")
            yield ProviderInfo(**provider.get_dict(), model=alias)
            extra_body = kwargs.copy()
            current_provider_api_key = None
            if isinstance(api_key, dict):
                current_provider_api_key = api_key.get(provider.get_parent())
            if not api_key:
                current_provider_api_key = AuthManager.load_api_key(provider)
            if current_provider_api_key:
                extra_body["api_key"] = current_provider_api_key
            if conversation is not None and hasattr(conversation, provider.__name__):
                extra_body["conversation"] = JsonConversation(**getattr(conversation, provider.__name__))
            try:
                response = provider.async_create_function(model, messages, **extra_body)
                if hasattr(response, "__aiter__"):
                    async for chunk in response:
                        if isinstance(chunk, JsonConversation):
                            if conversation is None:
                                conversation = JsonConversation()
                            setattr(conversation, provider.__name__, chunk.get_dict())
                            yield conversation
                        elif chunk:
                            yield chunk
                            if is_content(chunk):
                                started = True
                elif response:
                    response = await response
                    if response:
                        yield response
                        started = True
                if started:
                    return
            except Exception as e:
                exceptions[provider.__name__] = e
                debug.error(f"{provider.__name__}:", e)
                if started:
                    raise e
                yield e

        raise_exceptions(exceptions)

    create_function = create_completion
    async_create_function = create_async_generator

    def get_providers(self, ignored: list[str]) -> list[ProviderType]:
        providers = [p for p in self.providers if p.__name__ not in ignored]
        if self.shuffle:
            random.shuffle(providers)
        return providers

class RetryProvider(IterListProvider):
    def __init__(
        self,
        providers: List[Type[BaseProvider]],
        shuffle: bool = True,
        single_provider_retry: bool = False,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the BaseRetryProvider.
        Args:
            providers (List[Type[BaseProvider]]): List of providers to use.
            shuffle (bool): Whether to shuffle the providers list.
            single_provider_retry (bool): Whether to retry a single provider if it fails.
            max_retries (int): Maximum number of retries for a single provider.
        """
        super().__init__(providers, shuffle)
        self.single_provider_retry = single_provider_retry
        self.max_retries = max_retries

    def create_completion(
        self,
        model: str,
        messages: Messages,
        **kwargs,
    ) -> CreateResult:
        """
        Create a completion using available providers.
        Args:
            model (str): The model to be used for completion.
            messages (Messages): The messages to be used for generating completion.
        Yields:
            CreateResult: Tokens or results from the completion.
        Raises:
            Exception: Any exception encountered during the completion process.
        """
        if self.single_provider_retry:
            exceptions = {}
            started: bool = False
            provider = self.providers[0]
            self.last_provider = provider
            for attempt in range(self.max_retries):
                try:
                    if debug.logging:
                        print(f"Using {provider.__name__} provider (attempt {attempt + 1})")
                    response = provider.create_function(model, messages, **kwargs)
                    for chunk in response:
                        yield chunk
                        if is_content(chunk):
                            started = True
                    if started:
                        return
                except Exception as e:
                    exceptions[provider.__name__] = e
                    if debug.logging:
                        print(f"{provider.__name__}: {e.__class__.__name__}: {e}")
                    if started:
                        raise e
            raise_exceptions(exceptions)
        else:
            yield from super().create_completion(model, messages, **kwargs)

    async def create_async_generator(
        self,
        model: str,
        messages: Messages,
        **kwargs
    ) -> AsyncResult:
        exceptions = {}
        started = False

        if self.single_provider_retry:
            provider = self.providers[0]
            self.last_provider = provider
            for attempt in range(self.max_retries):
                try:
                    debug.log(f"Using {provider.__name__} provider (attempt {attempt + 1})")
                    response = provider.async_create_function(model, messages, **kwargs)
                    if hasattr(response, "__aiter__"):
                        async for chunk in response:
                            yield chunk
                            if is_content(chunk):
                                started = True
                    else:
                        response = await response
                        if response:
                            yield response
                            started = True
                    if started:
                        return
                except Exception as e:
                    exceptions[provider.__name__] = e
                    if debug.logging:
                        print(f"{provider.__name__}: {e.__class__.__name__}: {e}")
            raise_exceptions(exceptions)
        else:
            async for chunk in super().create_async_generator(model, messages, **kwargs):
                yield chunk
                
def raise_exceptions(exceptions: dict) -> None:
    """
    Raise a combined exception if any occurred during retries.

    Raises:
        RetryProviderError: If any provider encountered an exception.
        RetryNoProviderError: If no provider is found.
    """
    if exceptions:
        for provider_name, e in exceptions.items():
            if isinstance(e, (MissingAuthError, NoValidHarFileError)):
                raise e
        if len(exceptions) == 1:
            raise list(exceptions.values())[0]
        raise RetryProviderError("RetryProvider failed:\n" + "\n".join([
            f"{p}: {type(exception).__name__}: {exception}" for p, exception in exceptions.items()
        ])) from list(exceptions.values())[0]

    raise RetryNoProviderError("No content response from any provider. ")