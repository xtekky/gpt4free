from __future__ import annotations

import random

from ..typing import CreateResult
from .base_provider import BaseProvider, AsyncProvider
from ..debug import logging


class RetryProvider(AsyncProvider):
    __name__ = "RetryProvider"
    working               = True
    needs_auth            = False
    supports_stream       = True
    supports_gpt_35_turbo = False
    supports_gpt_4        = False

    def __init__(
        self,
        providers: list[type[BaseProvider]],
        shuffle: bool = True
    ) -> None:
        self.providers = providers
        self.shuffle = shuffle


    def create_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> CreateResult:
        if stream:
            providers = [provider for provider in self.providers if provider.supports_stream]
        else:
            providers = self.providers
        if self.shuffle:
            random.shuffle(providers)

        self.exceptions = {}
        started = False
        for provider in providers:
            try:
                if logging:
                    print(f"Using {provider.__name__} provider")
                for token in provider.create_completion(model, messages, stream, **kwargs):
                    yield token
                    started = True
                if started:
                    return
            except Exception as e:
                self.exceptions[provider.__name__] = e
                if logging:
                    print(f"{provider.__name__}: {e.__class__.__name__}: {e}")
                if started:
                    break

        self.raise_exceptions()

    async def create_async(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs
    ) -> str:
        providers = [provider for provider in self.providers]
        if self.shuffle:
            random.shuffle(providers)
        
        self.exceptions = {}
        for provider in providers:
            try:
                return await provider.create_async(model, messages, **kwargs)
            except Exception as e:
                self.exceptions[provider.__name__] = e
                if logging:
                    print(f"{provider.__name__}: {e.__class__.__name__}: {e}")
    
        self.raise_exceptions()
    
    def raise_exceptions(self):
        if self.exceptions:
            raise RuntimeError("\n".join(["All providers failed:"] + [
                f"{p}: {self.exceptions[p].__class__.__name__}: {self.exceptions[p]}" for p in self.exceptions
            ]))
        
        raise RuntimeError("No provider found")