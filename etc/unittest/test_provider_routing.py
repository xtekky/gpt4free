from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from g4f.providers.base_provider import AsyncGeneratorProvider
from g4f.providers.provider_router import ProviderRouter
from g4f.providers.retry_provider import ProviderCircuitBreaker, RotatedProvider
from g4f.tools.auth import AuthManager
from g4f.errors import MissingAuthError


class RoutingProvider(AsyncGeneratorProvider):
    working = True
    needs_auth = False
    supports_native_tools = False
    default_model = "test-model"
    models = ["test-model"]
    live = 0

    @classmethod
    def get_parent(cls) -> str:
        return cls.__name__

    @classmethod
    async def create_async_generator(cls, model, messages, **kwargs):
        yield "ok"


class FreeProvider(RoutingProvider):
    pass


class ToolProvider(RoutingProvider):
    supports_native_tools = True


class KeyedProvider(RoutingProvider):
    models_needs_auth = True


class UnknownProvider(RoutingProvider):
    models_needs_auth = True


class FakeAudio(RoutingProvider):
    models_needs_auth = True
    default_model = "voice"
    models = ["voice"]


class CachedProvider(RoutingProvider):
    models_needs_auth = True
    cache_file: Path = None

    @classmethod
    def get_cache_file(cls) -> Path:
        return cls.cache_file


class DiscoveredProvider(RoutingProvider):
    models_needs_auth = True
    models = []
    model_aliases = {"delta-alias": "delta"}

    @classmethod
    def get_models(cls, api_key=None, **kwargs):
        return ["delta", "epsilon"]


class DynamicFreeProvider(RoutingProvider):
    default_model = None
    models = []

    @classmethod
    def get_models(cls, **kwargs):
        return ["omega"]


class FailingProvider(AsyncGeneratorProvider):
    working = True

    @classmethod
    async def create_async_generator(cls, model, messages, **kwargs):
        raise RuntimeError("provider down")
        yield  # pragma: no cover


class MissingAuthProvider(AsyncGeneratorProvider):
    working = True

    @classmethod
    async def create_async_generator(cls, model, messages, **kwargs):
        raise MissingAuthError("login required")
        yield  # pragma: no cover


class TestProviderAuthDetection(unittest.TestCase):
    def test_environment_key_is_detected(self):
        with patch.dict(os.environ, {"KEYEDPROVIDER_API_KEY": "secret"}, clear=False):
            status = AuthManager.get_auth_status(KeyedProvider)
        self.assertTrue(status.available)
        self.assertTrue(status.configured)
        self.assertEqual(status.source, "env")
        self.assertEqual(status.api_key, "secret")

    def test_request_key_is_detected(self):
        status = AuthManager.get_auth_status(
            KeyedProvider, api_key={"KeyedProvider": "secret"}
        )
        self.assertTrue(status.configured)
        self.assertEqual(status.source, "request")
        self.assertEqual(status.api_key, "secret")

    def test_free_provider_needs_no_auth(self):
        status = AuthManager.get_auth_status(FreeProvider)
        self.assertTrue(status.available)
        self.assertEqual(status.source, "free")
        self.assertFalse(status.configured)

    def test_missing_auth_is_unknown(self):
        with patch.dict(os.environ, {}, clear=True):
            status = AuthManager.get_auth_status(UnknownProvider)
        self.assertFalse(status.available)
        self.assertEqual(status.source, "unknown")

    def test_cached_auth_file_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_file = Path(directory) / "auth_CachedProvider.json"
            cache_file.write_text("{}")
            CachedProvider.cache_file = cache_file
            try:
                with patch.dict(os.environ, {}, clear=True):
                    status = AuthManager.get_auth_status(CachedProvider)
            finally:
                CachedProvider.cache_file = None
        self.assertTrue(status.available)
        self.assertEqual(status.source, "cache")

    def test_g4f_keys_are_not_provider_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            status = AuthManager.get_auth_status(
                KeyedProvider, api_key="g4f_not-a-provider-key"
            )
        self.assertFalse(status.configured)


class TestProviderRanking(unittest.TestCase):
    def setUp(self):
        ProviderCircuitBreaker.reset()

    def tearDown(self):
        ProviderCircuitBreaker.reset()

    def test_configured_provider_beats_free_and_unknown(self):
        with patch.dict(os.environ, {"KEYEDPROVIDER_API_KEY": "secret"}, clear=False):
            ranked = ProviderRouter.rank(
                [UnknownProvider, FreeProvider, KeyedProvider], api_key=None
            )
        self.assertEqual(
            [provider.__name__ for provider in ranked],
            ["KeyedProvider", "FreeProvider", "UnknownProvider"],
        )

    def test_native_tool_provider_is_tried_first(self):
        ranked = ProviderRouter.rank(
            [FreeProvider, ToolProvider], has_tools=True, api_key=None
        )
        self.assertEqual(
            [provider.__name__ for provider in ranked],
            ["ToolProvider", "FreeProvider"],
        )

    def test_cooling_provider_moves_to_the_back(self):
        for _ in range(ProviderCircuitBreaker.MAX_CONSECUTIVE_FAILURES):
            ProviderCircuitBreaker.record_failure("ToolProvider")
        ranked = ProviderRouter.rank(
            [ToolProvider, FreeProvider], has_tools=True, api_key=None
        )
        self.assertEqual(
            [provider.__name__ for provider in ranked],
            ["FreeProvider", "ToolProvider"],
        )

    def test_ignored_provider_is_removed(self):
        ranked = ProviderRouter.rank(
            [FreeProvider, ToolProvider],
            has_tools=True,
            api_key=None,
            ignored=["ToolProvider"],
        )
        self.assertEqual([provider.__name__ for provider in ranked], ["FreeProvider"])


class TestProviderModelMatching(unittest.TestCase):
    def test_exact_alias_and_normalized_matches(self):
        class MatchProvider(RoutingProvider):
            models = ["real-model", "foo-bar"]
            model_aliases = {"alias": "real-model"}

        self.assertEqual(
            ProviderRouter.match_model(MatchProvider, "alias", MatchProvider.models),
            "real-model",
        )
        self.assertEqual(
            ProviderRouter.match_model(MatchProvider, "FOO_BAR", MatchProvider.models),
            "foo-bar",
        )
        self.assertIsNone(
            ProviderRouter.match_model(MatchProvider, "other", MatchProvider.models)
        )

    def test_models_aliases_attribute_is_used(self):
        class AliasedProvider(RoutingProvider):
            models = ["claude-sonnet-4-20250514"]
            models_aliases = {"claude-4-sonnet": "claude-sonnet-4-20250514"}

        self.assertEqual(
            ProviderRouter.match_model(AliasedProvider, "claude-4-sonnet"),
            "claude-sonnet-4-20250514",
        )

    def test_normalized_alias_key_is_used(self):
        class AliasedProvider(RoutingProvider):
            models = ["claude-sonnet-4-20250514"]
            models_aliases = {"claude-4.5-sonnet": "claude-sonnet-4-20250514"}

        self.assertEqual(
            ProviderRouter.match_model(AliasedProvider, "claude-4-5-sonnet"),
            "claude-sonnet-4-20250514",
        )

    def test_empty_alias_uses_provider_default(self):
        class DefaultAliasProvider(RoutingProvider):
            models = ["auto"]
            default_model = "auto"
            model_aliases = {"gpt-4o": ""}

        self.assertEqual(
            ProviderRouter.match_model(DefaultAliasProvider, "gpt-4o"), "auto"
        )

    def test_alias_target_must_be_advertised(self):
        # Media-only subclasses inherit text aliases; discovery must not send
        # a text request to them.
        class ImageOnlyProvider(RoutingProvider):
            models = ["auto"]
            default_model = "auto"
            model_aliases = {"gpt-4o": "openai-fast"}

        self.assertIsNone(
            ProviderRouter.match_model(ImageOnlyProvider, "gpt-4o")
        )


class TestProviderDiscovery(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ProviderRouter.reset_cache()

    async def test_configured_provider_is_discovered(self):
        with patch.dict(
            os.environ, {"DISCOVEREDPROVIDER_API_KEY": "secret"}, clear=False
        ):
            found = await ProviderRouter.discover(
                "delta-alias", providers=[DiscoveredProvider]
            )
        self.assertEqual(found, [DiscoveredProvider])
        self.assertEqual(
            DiscoveredProvider.model_aliases.get("delta-alias"), "delta"
        )

    async def test_unconfigured_provider_is_not_probed(self):
        with patch.dict(os.environ, {}, clear=True):
            found = await ProviderRouter.discover(
                "delta", providers=[DiscoveredProvider]
            )
        self.assertEqual(found, [])

    async def test_broad_discovery_probes_free_provider(self):
        found = await ProviderRouter.discover(
            "omega", providers=[DynamicFreeProvider], broad=True
        )
        self.assertEqual(found, [DynamicFreeProvider])

    def test_media_only_provider_is_not_auto_selected(self):
        with patch.dict(
            os.environ,
            {
                "KEYEDPROVIDER_API_KEY": "secret",
                "FAKEAUDIO_API_KEY": "secret",
            },
            clear=True,
        ):
            configured = ProviderRouter.authed_providers(
                providers=[KeyedProvider, FakeAudio]
            )
        self.assertEqual(configured, [KeyedProvider])


class TestRotatedProviderCircuitBreaker(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ProviderCircuitBreaker.reset()

    def tearDown(self):
        ProviderCircuitBreaker.reset()

    async def test_failure_is_recorded(self):
        provider = RotatedProvider([FailingProvider], shuffle=False)
        with self.assertRaises(RuntimeError):
            async for _ in provider.create_async_generator(
                "test-model", [{"role": "user", "content": "hi"}]
            ):
                pass
        self.assertEqual(ProviderCircuitBreaker._failures.get("FailingProvider"), 1)

    async def test_missing_auth_is_not_counted_as_provider_failure(self):
        provider = RotatedProvider([MissingAuthProvider], shuffle=False)
        with self.assertRaises(MissingAuthError):
            async for _ in provider.create_async_generator(
                "test-model", [{"role": "user", "content": "hi"}]
            ):
                pass
        self.assertNotIn("MissingAuthProvider", ProviderCircuitBreaker._failures)

    async def test_success_clears_previous_failures(self):
        ProviderCircuitBreaker.record_failure("FreeProvider")
        provider = RotatedProvider([FreeProvider], shuffle=False)
        result = [
            chunk
            async for chunk in provider.create_async_generator(
                "test-model", [{"role": "user", "content": "hi"}]
            )
        ]
        self.assertTrue(any(chunk == "ok" for chunk in result))
        self.assertNotIn("FreeProvider", ProviderCircuitBreaker._failures)
