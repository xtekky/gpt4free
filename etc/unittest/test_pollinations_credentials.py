from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from g4f.Provider.Pollinations import Pollinations
from g4f.tools.auth import AuthManager


class TestPollinationsStoredCredentials(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.home = Path(self.directory.name)
        (self.home / ".pollinations").mkdir()

    def tearDown(self):
        self.directory.cleanup()

    def write_credentials(self, filename: str, data) -> None:
        (self.home / ".pollinations" / filename).write_text(
            json.dumps(data), encoding="utf-8"
        )

    def test_reads_polli_credentials_file(self):
        self.write_credentials(
            "credentials.json", {"apiKey": "sk_test", "keyType": "sk"}
        )
        with patch.dict(os.environ, {"HOME": str(self.home)}, clear=True):
            self.assertEqual(Pollinations.load_stored_api_key(), "sk_test")

    def test_uses_staging_credentials_when_staging(self):
        self.write_credentials("credentials.json", {"apiKey": "sk_live"})
        self.write_credentials("credentials.staging.json", {"apiKey": "sk_staging"})
        with patch.dict(
            os.environ,
            {"HOME": str(self.home), "POLLINATIONS_ENV": "staging"},
            clear=True,
        ):
            self.assertEqual(Pollinations.load_stored_api_key(), "sk_staging")

    def test_missing_or_invalid_credentials_are_ignored(self):
        with patch.dict(os.environ, {"HOME": str(self.home)}, clear=True):
            self.assertIsNone(Pollinations.load_stored_api_key())
        self.write_credentials("credentials.json", {"keyType": "sk"})
        with patch.dict(os.environ, {"HOME": str(self.home)}, clear=True):
            self.assertIsNone(Pollinations.load_stored_api_key())
        (self.home / ".pollinations" / "credentials.json").write_text("{")
        with patch.dict(os.environ, {"HOME": str(self.home)}, clear=True):
            self.assertIsNone(Pollinations.load_stored_api_key())

    def test_auth_manager_falls_back_to_stored_credentials(self):
        self.write_credentials("credentials.json", {"apiKey": "sk_stored"})
        with patch.dict(os.environ, {"HOME": str(self.home)}, clear=True):
            self.assertEqual(AuthManager.load_api_key(Pollinations), "sk_stored")

    def test_environment_key_takes_precedence(self):
        self.write_credentials("credentials.json", {"apiKey": "sk_stored"})
        with patch.dict(
            os.environ,
            {"HOME": str(self.home), "POLLINATIONS_API_KEY": "sk_env"},
            clear=True,
        ):
            self.assertEqual(AuthManager.load_api_key(Pollinations), "sk_env")


if __name__ == "__main__":
    unittest.main()
