from __future__ import annotations

import asyncio
import random
from ..typing import CreateResult, Messages
from ..base_provider import BaseRetryProvider
from .. import debug
from ..errors import RetryProviderError, RetryNoProviderError


class RetryProvider(BaseRetryProvider):
    def create_completion(
        self,
        model: str,
        messages: Messages,
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
        if self.exceptions:
            raise RetryProviderError("RetryProvider failed:\n" + "\n".join([
                f"{p}: {exception.__class__.__name__}: {exception}" for p, exception in self.exceptions.items()
            ]))
        
        raise RetryNoProviderError("No provider found")