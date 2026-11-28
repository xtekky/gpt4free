from __future__ import annotations

import asyncio
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from g4f.Provider.needs_auth.Antigravity import (
    Antigravity,
    AntigravityAccountPool,
    AntigravityAuthManager,
    AntigravityProvider,
    AntigravityQuotaError,
    get_auto_model_candidates,
    load_antigravity_accounts,
    parse_antigravity_quota_error,
)

antigravity_module = importlib.import_module("g4f.Provider.needs_auth.Antigravity")


async def _collect(generator):
    return [chunk async for chunk in generator]


class TestAntigravityModels(unittest.TestCase):
    def test_auto_uses_first_midsize_gemini_then_claude(self):
        models = [
            "gemini-3-pro-high",
            "claude-sonnet-4-6",
            "gemini-3.5-flash-low",
            "gemini-3.1-flash-lite",
        ]
        self.assertEqual(
            get_auto_model_candidates(models),
            ["gemini-3.5-flash-low", "claude-sonnet-4-6"],
        )

    def test_auto_falls_back_to_builtin_models(self):
        candidates = get_auto_model_candidates(["gemini-3-pro-high"])
        self.assertEqual(candidates[0], "gemini-3-flash")
        self.assertIn("claude", candidates[1])

    def test_quota_error_parsing(self):
        error = parse_antigravity_quota_error(
            429,
            json.dumps({
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [{"retryDelay": "45s"}],
                }
            }),
        )
        self.assertIsInstance(error, AntigravityQuotaError)
        self.assertEqual(error.retry_after, 45)
        self.assertIsNone(parse_antigravity_quota_error(500, "internal error"))


class TestAntigravityAccounts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.accounts_dir = self.root / "accounts"
        self.accounts_dir.mkdir()
        for name, email in (("work", "work@example.com"), ("personal", "me@example.com")):
            (self.accounts_dir / f"{name}.json").write_text(json.dumps({
                "refresh_token": f"token-{name}",
                "email": email,
            }))

    def _load_with_isolated_paths(self, extra_env=None):
        env = {
            "ANTIGRAVITY_ACCOUNTS_DIR": str(self.accounts_dir),
            **(extra_env or {}),
        }
        with patch.object(
            AntigravityAuthManager,
            "get_cache_file",
            return_value=self.root / "missing-cache.json",
        ), patch.object(
            antigravity_module,
            "get_antigravity_oauth_creds_path",
            return_value=self.root / "missing-oauth.json",
        ), patch.dict(os.environ, env, clear=False):
            return load_antigravity_accounts(os.environ)

    def test_loads_multiple_account_files_and_env_accounts(self):
        env_accounts = json.dumps([
            {"refresh_token": "env-token", "email": "env@example.com"}
        ])
        accounts = self._load_with_isolated_paths(
            {"ANTIGRAVITY_SERVICE_ACCOUNT": env_accounts}
        )
        self.assertEqual(len(accounts), 3)
        self.assertEqual(
            sorted(account["name"] for account in accounts),
            ["env@example.com", "personal", "work"],
        )

    def test_pool_switches_account_when_quota_reached(self):
        managers = [
            AntigravityAuthManager(
                env={}, credentials={"refresh_token": "first"}, name="first"
            ),
            AntigravityAuthManager(
                env={}, credentials={"refresh_token": "second"}, name="second"
            ),
        ]
        pool = AntigravityAccountPool(env={}, managers=managers)
        attempts = []

        async def fake_stream(self, model, messages, **kwargs):
            attempts.append((self.auth_manager.name, model))
            if self.auth_manager.name == "first":
                raise AntigravityQuotaError("quota reached")
            yield "ok"

        with patch.object(Antigravity, "_get_account_pool", return_value=pool), patch.object(
            AntigravityProvider, "stream_content", fake_stream
        ):
            chunks = asyncio.run(
                _collect(Antigravity.create_async_generator("gemini-3-flash", []))
            )

        self.assertEqual(chunks, ["ok"])
        self.assertEqual(attempts, [("first", "gemini-3-flash"), ("second", "gemini-3-flash")])
        self.assertTrue(managers[0].is_quota_exhausted("gemini"))

    def test_auto_switches_to_claude_when_all_gemini_accounts_are_limited(self):
        managers = [
            AntigravityAuthManager(env={}, credentials={"refresh_token": "a"}, name="a"),
            AntigravityAuthManager(env={}, credentials={"refresh_token": "b"}, name="b"),
        ]
        pool = AntigravityAccountPool(env={}, managers=managers)
        attempts = []

        async def fake_stream(self, model, messages, **kwargs):
            attempts.append((self.auth_manager.name, model))
            if "gemini" in model:
                raise AntigravityQuotaError("gemini quota reached")
            yield model

        with patch.object(Antigravity, "_get_account_pool", return_value=pool), patch.object(
            Antigravity, "models", []
        ), patch.object(AntigravityProvider, "stream_content", fake_stream):
            chunks = asyncio.run(_collect(Antigravity.create_async_generator("auto", [])))

        self.assertEqual(chunks, ["claude-sonnet-4-6"])
        self.assertEqual(
            attempts,
            [
                ("a", "gemini-3-flash"),
                ("b", "gemini-3-flash"),
                ("a", "claude-sonnet-4-6"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
