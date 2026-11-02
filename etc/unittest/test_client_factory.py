from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from g4f.client.factory import AbstractClientFactory


class TestLiveProviderRegistry(unittest.TestCase):
    """The live provider list is fetched while ``g4f.Provider`` is imported.

    It used to be an unbounded ``requests.get`` and an exception escaped from
    ``import g4f.gui``, so offline devices either hung on the splash screen
    or crashed on startup.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "providers.json"
        self._saved = AbstractClientFactory._live_providers
        AbstractClientFactory._live_providers = {}

    def tearDown(self):
        AbstractClientFactory._live_providers = self._saved
        self.tmp.cleanup()

    def test_offline_uses_fallback_without_raising(self):
        with patch(
            "g4f.client.factory.requests.get",
            side_effect=requests.ConnectionError("offline"),
        ) as get:
            data = AbstractClientFactory._load_live_providers(self.path)
        self.assertIn("default", data["providers"])
        self.assertEqual(data["defaultModels"]["default"], "auto")
        self.assertEqual(get.call_args.kwargs.get("timeout"), 10)
        self.assertFalse(self.path.exists())

    def test_cached_registry_skips_network(self):
        self.path.write_text(
            json.dumps({"providers": {"cached": {}}, "defaultModels": {}})
        )
        with patch("g4f.client.factory.requests.get") as get:
            data = AbstractClientFactory._load_live_providers(self.path)
        self.assertEqual(list(data["providers"]), ["cached"])
        get.assert_not_called()

    def test_downloaded_registry_is_cached(self):
        payload = {"providers": {"live": {"label": "live"}}, "defaultModels": {}}
        response = Mock()
        response.json.return_value = payload
        with patch("g4f.client.factory.requests.get", return_value=response):
            data = AbstractClientFactory._load_live_providers(self.path)
        self.assertEqual(data, payload)
        self.assertEqual(json.loads(self.path.read_text()), payload)

    def test_default_provider_resolves_offline(self):
        # Force the live-registry branch: while g4f.Provider is still being
        # imported its lazy ``convert`` map cannot resolve "default", which is
        # exactly the path that reached the network during ``import g4f.gui``.
        with patch(
            "g4f.client.factory.requests.get",
            side_effect=requests.ConnectionError("offline"),
        ), patch("g4f.client.factory.ProviderUtils") as utils:
            utils.convert = {}
            utils.get_by_label.side_effect = ValueError("not a label")
            provider = AbstractClientFactory.create_provider(None, "default")
        self.assertTrue(provider.base_url.startswith("https://g4f.space"))
        self.assertEqual(provider.default_model, "auto")


if __name__ == "__main__":
    unittest.main()
