import unittest
from typing import Type
import asyncio

from g4f.models import __models__
from g4f.providers.base_provider import BaseProvider, ProviderModelMixin
from g4f.errors import MissingRequirementsError, MissingAuthError

class TestProviderHasModel(unittest.TestCase):
    cache: dict = {}

    def test_provider_has_model(self):
        for model, providers in __models__.values():
            for provider in providers:
                if provider.needs_auth:
                    continue
                if issubclass(provider, ProviderModelMixin):
                    provider.get_models() # Update models
                    if model.name in provider.model_aliases:
                        model_name = provider.model_aliases[model.name]
                    else:
                        model_name = model.name
                        self.provider_has_model(provider, model_name)

    def provider_has_model(self, provider: Type[BaseProvider], model: str):
        if provider.__name__ not in self.cache:
            try:
                self.cache[provider.__name__] = provider.get_models()
            except (MissingRequirementsError, MissingAuthError):
                return
        if self.cache[provider.__name__]:
            self.assertIn(model, self.cache[provider.__name__], provider.__name__)

    def test_all_providers_working(self):
        for model, providers in __models__.values():
            for provider in providers:
                self.assertTrue(provider.working, f"{provider.__name__} in {model.name}")