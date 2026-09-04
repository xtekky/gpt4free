"""
Unit tests for AsyncAuthedProvider auth-failure retry behaviour.

Covers the fix for #3515: when a cached access token is rejected
(MissingAuthError), the provider must invalidate the stale cache file and
clear in-memory auth state so the automatic re-login fetches fresh
credentials instead of reusing (and failing on) the rejected ones.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from g4f.errors import MissingAuthError
from g4f.providers.response import AuthResult

from .mocks import RetryAuthedProviderMock

DEFAULT_MESSAGES = [{"role": "user", "content": "Hello"}]


class TestAuthRetry(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        RetryAuthedProviderMock._api_key = None
        RetryAuthedProviderMock._headers = None
        RetryAuthedProviderMock._cookies = None
        RetryAuthedProviderMock._expires = None

    async def test_stale_cache_is_invalidated_and_retry_succeeds(self):
        """A rejected cached token triggers reset_auth and a fresh re-login."""
        provider = RetryAuthedProviderMock
        cache_file = provider.get_cache_file()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w") as f:
            json.dump(AuthResult(api_key="stale-token").get_dict(), f)
        self.assertTrue(cache_file.exists())

        try:
            reset_calls = []
            original_reset = provider.reset_auth

            def tracking_reset():
                reset_calls.append(True)
                original_reset()

            with patch.object(provider, "reset_auth", tracking_reset):
                chunks = [
                    chunk
                    async for chunk in provider.create_async_generator(
                        "model", DEFAULT_MESSAGES
                    )
                ]

            self.assertEqual(chunks, ["Hello"])
            # reset_auth was invoked to drop the stale credentials.
            self.assertEqual(reset_calls, [True])
            # The stale cache file was removed and a fresh one persisted.
            self.assertTrue(cache_file.exists())
            with cache_file.open("r") as f:
                saved = json.load(f)
            self.assertEqual(saved.get("api_key"), "fresh-token")
        finally:
            if cache_file.exists():
                cache_file.unlink()

    async def test_reset_auth_deletes_cache_file(self):
        """reset_auth removes the persisted auth cache file."""
        provider = RetryAuthedProviderMock
        cache_file = provider.get_cache_file()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w") as f:
            json.dump(AuthResult(api_key="stale-token").get_dict(), f)

        provider.reset_auth()
        self.assertFalse(cache_file.exists())
        # In-memory auth state is cleared too.
        self.assertIsNone(provider._api_key)
        self.assertIsNone(provider._expires)

    async def test_missing_cache_triggers_login(self):
        """With no cache file, a fresh login is performed directly."""
        provider = RetryAuthedProviderMock
        cache_file = provider.get_cache_file()
        if cache_file.exists():
            cache_file.unlink()

        chunks = [
            chunk
            async for chunk in provider.create_async_generator(
                "model", DEFAULT_MESSAGES
            )
        ]
        self.assertEqual(chunks, ["Hello"])

    async def test_persistent_auth_failure_propagates(self):
        """If the re-login still fails, the error propagates (no silent loop)."""

        class AlwaysFailingProvider(RetryAuthedProviderMock):
            parent = "AlwaysFailingProvider"

            @classmethod
            async def create_authed(cls, model, messages, auth_result, **kwargs):
                raise MissingAuthError("Access token is not valid")
                yield  # pragma: no cover

        provider = AlwaysFailingProvider
        cache_file = provider.get_cache_file()
        if cache_file.exists():
            cache_file.unlink()
        try:
            with self.assertRaises(MissingAuthError):
                [
                    chunk
                    async for chunk in provider.create_async_generator(
                        "model", DEFAULT_MESSAGES
                    )
                ]
        finally:
            if cache_file.exists():
                cache_file.unlink()


if __name__ == "__main__":
    unittest.main()
