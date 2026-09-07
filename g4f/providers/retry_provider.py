from __future__ import annotations

import time
import os
import random

from ..typing import Dict, Type, List, Messages, AsyncResult
from .types import BaseProvider, BaseRetryProvider, ProviderType
from .response import ProviderInfo, JsonConversation, is_content
from .base_provider import get_async_provider_method
from .. import debug
from ..tools.auth import AuthManager
from ..config import AppConfig
from ..errors import RetryProviderError, RetryNoProviderError


def _resolve_model(provider: Type[BaseProvider], model: str) -> str:
    alias = model or getattr(provider, "default_model", None)
    if getattr(provider, "model_aliases", None):
        alias = provider.model_aliases.get(model, model)
    if isinstance(alias, list):
        alias = random.choice(alias)
    return alias


def _prepare_provider_kwargs(
    provider: Type[BaseProvider],
    api_key,
    conversation: JsonConversation,
    kwargs: dict,
) -> dict:
    extra_body = kwargs.copy()
    current_api_key = (
        api_key.get(provider.get_parent()) if isinstance(api_key, dict) else api_key
    )
    if not current_api_key or AppConfig.disable_custom_api_key:
        current_api_key = AuthManager.load_api_key(provider)
    if current_api_key:
        extra_body["api_key"] = current_api_key
    if conversation is not None and hasattr(conversation, provider.__name__):
        extra_body["conversation"] = JsonConversation(
            **getattr(conversation, provider.__name__)
        )
    return extra_body


class RotatedProvider(BaseRetryProvider):
    """
    A provider that rotates through a list of providers, attempting one provider per
    request and advancing to the next one upon failure. This distributes load and
    retries across multiple providers in a round-robin fashion.
    """

    def __init__(
        self, providers: List[Type[BaseProvider]], shuffle: bool = True
    ) -> None:
        """
        Initialize the RotatedProvider.
        Args:
            providers (List[Type[BaseProvider]]): A non-empty list of providers to rotate through.
            shuffle (bool): If True, shuffles the provider list once at initialization
                            to randomize the rotation order.
        """
        if not isinstance(providers, list) or len(providers) == 0:
            raise ValueError("RotatedProvider requires a non-empty list of providers.")

        self.providers = providers
        if shuffle:
            random.shuffle(self.providers)

        self.current_index = 0
        self.last_provider: Type[BaseProvider] = None

    def _get_current_provider(self) -> Type[BaseProvider]:
        """Gets the provider at the current index."""
        p = self.providers[self.current_index]
        if isinstance(p, str):
            from ..Provider import __getattr__

            p = __getattr__(p)
        return p

    def _rotate_provider(self) -> None:
        """Rotates to the next provider in the list."""
        self.current_index = (self.current_index + 1) % len(self.providers)
        # new_provider_name = self.providers[self.current_index].__name__
        # debug.log(f"Rotated to next provider: {new_provider_name}")

    async def create_async_generator(
        self,
        model: str,
        messages: Messages,
        ignored: list[str] = [],
        api_key: str = None,
        conversation: JsonConversation = None,
        **kwargs,
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

            alias = _resolve_model(provider, model)

            debug.log(f"Attempting provider: {provider.__name__} with model: {alias}")
            yield ProviderInfo(**provider.get_dict(), model=alias)

            extra_body = _prepare_provider_kwargs(
                provider, api_key, conversation, kwargs
            )

            try:
                method = get_async_provider_method(provider)
                response = method(model=alias, messages=messages, **extra_body)
                started = False
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
                if started:
                    provider.live += 1
                    return  # Success
            except Exception as e:
                provider.live -= 1
                exceptions[provider.__name__] = e
                debug.error(f"{provider.__name__} failed: {e}")

        raise_exceptions(exceptions)


class ProviderCircuitBreaker:
    """Tracks provider failure states to avoid repeatedly querying unavailable providers."""
    _failures: Dict[str, int] = {}
    _cooldowns: Dict[str, float] = {}
    COOLDOWN_SECONDS: float = 60.0
    MAX_CONSECUTIVE_FAILURES: int = 3

    @classmethod
    def is_available(cls, provider_name: str) -> bool:
        if os.environ.get("G4F_DISABLE_CIRCUIT_BREAKER", "").lower() in ("1", "true"):
            return True
        if provider_name in cls._cooldowns:
            if time.time() < cls._cooldowns[provider_name]:
                return False
            cls._cooldowns.pop(provider_name, None)
            cls._failures[provider_name] = cls.MAX_CONSECUTIVE_FAILURES - 1
        return True

    @classmethod
    def record_failure(cls, provider_name: str, cooldown: Optional[float] = None):
        cls._failures[provider_name] = cls._failures.get(provider_name, 0) + 1
        if cls._failures[provider_name] >= cls.MAX_CONSECUTIVE_FAILURES:
            cooldown_time = cooldown or cls.COOLDOWN_SECONDS
            cls._cooldowns[provider_name] = time.time() + cooldown_time
            debug.log(f"Provider {provider_name} entered cooldown for {cooldown_time}s")

    @classmethod
    def record_success(cls, provider_name: str):
        cls._failures.pop(provider_name, None)
        cls._cooldowns.pop(provider_name, None)

    @classmethod
    def reset(cls):
        cls._failures.clear()
        cls._cooldowns.clear()


class IterListProvider(BaseRetryProvider):
    def __init__(
        self, providers: List[Type[BaseProvider]] = [], shuffle: bool = True
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

    async def create_async_generator(
        self,
        model: str,
        messages: Messages,
        ignored: list[str] = [],
        api_key: str = None,
        conversation: JsonConversation = None,
        **kwargs,
    ) -> AsyncResult:
        exceptions = {}
        started: bool = False

        for provider in self.get_providers(ignored):
            self.last_provider = provider
            alias = _resolve_model(provider, model)
            debug.log(f"Using {provider.__name__} provider with model {alias}")
            yield ProviderInfo(**provider.get_dict(), model=alias)
            extra_body = _prepare_provider_kwargs(
                provider, api_key, conversation, kwargs
            )
            try:
                method = get_async_provider_method(provider)
                response = method(model=alias, messages=messages, **extra_body)
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
                if started:
                    ProviderCircuitBreaker.record_success(provider.__name__)
                    return
            except Exception as e:
                ProviderCircuitBreaker.record_failure(provider.__name__)
                exceptions[provider.__name__] = e
                debug.error(f"{provider.__name__}:", e)
                if started:
                    raise e

        raise_exceptions(exceptions)

    def get_providers(self, ignored: list[str] = []) -> list[ProviderType]:
        resolved_providers = []
        from ..Provider import __getattr__

        for p in self.providers:
            if isinstance(p, str):
                try:
                    p = __getattr__(p)
                except AttributeError:
                    continue
            if getattr(p, "__name__", "") not in ignored:
                resolved_providers.append(p)

        if self.shuffle:
            random.shuffle(resolved_providers)

        available_providers = []
        cooling_down_providers = []
        for p in resolved_providers:
            p_name = getattr(p, "__name__", str(p))
            if ProviderCircuitBreaker.is_available(p_name):
                available_providers.append(p)
            else:
                cooling_down_providers.append(p)

        return available_providers + cooling_down_providers


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

    async def create_async_generator(
        self, model: str, messages: Messages, **kwargs
    ) -> AsyncResult:
        exceptions = {}
        started = False

        if self.single_provider_retry:
            providers = self.get_providers()
            if not providers:
                raise RetryNoProviderError("No providers available")
            provider = providers[0]
            self.last_provider = provider
            for attempt in range(self.max_retries):
                try:
                    debug.log(
                        f"Using {provider.__name__} provider (attempt {attempt + 1})"
                    )
                    method = get_async_provider_method(provider)
                    response = method(model=model, messages=messages, **kwargs)
                    async for chunk in response:
                        yield chunk
                        if is_content(chunk):
                            started = True
                    if started:
                        return
                except Exception as e:
                    exceptions[provider.__name__] = e
                    if debug.logging:
                        print(f"{provider.__name__}: {e.__class__.__name__}: {e}")
            raise_exceptions(exceptions)
        else:
            async for chunk in super().create_async_generator(
                model, messages, **kwargs
            ):
                yield chunk


def raise_exceptions(exceptions: dict) -> None:
    """
    Raise a combined exception if any occurred during retries.

    Raises:
        RetryProviderError: If any provider encountered an exception.
        RetryNoProviderError: If no provider is found.
    """
    if exceptions:
        if len(exceptions) == 1:
            raise list(exceptions.values())[0]
        raise RetryProviderError(
            "RetryProvider failed:\n"
            + "\n".join(
                [
                    f"{p}: {type(exception).__name__}: {exception}"
                    for p, exception in exceptions.items()
                ]
            )
        ) from list(exceptions.values())[0]

    raise RetryNoProviderError("No content response from any provider. ")
