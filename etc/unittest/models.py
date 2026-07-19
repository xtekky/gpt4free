import unittest
from typing import Type
from requests.exceptions import RequestException

from g4f.Provider import __getattr__
from g4f.models import __models__
from g4f.providers.base_provider import BaseProvider, ProviderModelMixin
from g4f.errors import MissingRequirementsError, MissingAuthError

class TestProviderHasModel(unittest.TestCase):
    cache: dict = {}

    def test_provider_has_model(self):
        for model, providers in __models__.values():
            for provider in providers:
                if isinstance(provider, str):
                    try:
                        provider = __getattr__(provider)
                    except AttributeError:
                        continue
                if provider is None:
                    continue
                if getattr(provider, "needs_auth", False):
                    continue
                if issubclass(provider, ProviderModelMixin):
                    try:
                        provider.get_models(timeout=5) # Update models
                        if provider.model_aliases and model.name in provider.model_aliases:
                            model_name = provider.model_aliases[model.name]
                        else:
                            model_name = model.get_long_name()
                            self.provider_has_model(provider, model_name)
                    except RequestException:
                        continue

    def provider_has_model(self, provider: Type[BaseProvider], model: str):
        if provider.__name__ not in self.cache:
            try:
                self.cache[provider.__name__] = list(provider.get_models())
            except (MissingRequirementsError, MissingAuthError):
                return
        if self.cache[provider.__name__]:
            if not provider.model_aliases or model not in provider.model_aliases:
                self.assertIn(model, self.cache[provider.__name__], provider.__name__)

    def test_all_providers_working(self):
        for model, providers in __models__.values():
            for provider in providers:
                if isinstance(provider, str):
                    try:
                        provider = __getattr__(provider)
                    except AttributeError:
                        continue
                if provider is None:
                    continue
                self.assertTrue(provider.working, f"{provider.__name__} in {model.name}")