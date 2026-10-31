"""Tests for the local pi-freeflow provider."""

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from g4f.Provider import ProviderLoader
from g4f.Provider.local.PiFreeflow import (
    DEFAULT_BASE_URL,
    Freeflow,
    PiFreeflow,
    get_api_key,
    get_base_url,
    get_token_file,
)
from g4f.Provider.template import OpenaiTemplate


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class TestPiFreeflowSettings(unittest.TestCase):
    def test_default_base_url(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_base_url(), DEFAULT_BASE_URL)

    def test_base_url_from_environment(self):
        with patch.dict(
            os.environ, {"PI_FREEFLOW_BASE_URL": "http://freeflow.test:9000/v1"}
        ):
            self.assertEqual(
                get_base_url(), "http://freeflow.test:9000/v1"
            )
        with patch.dict(
            os.environ, {"FREEFLOW_BASE_URL": "http://freeflow.test:9000/"}
        ):
            self.assertEqual(
                get_base_url(), "http://freeflow.test:9000/v1"
            )

    def test_base_url_from_host_and_port(self):
        with patch.dict(
            os.environ,
            {"PI_FREEFLOW_HOST": "freeflow.test", "PI_FREEFLOW_PORT": "9999"},
            clear=True,
        ):
            self.assertEqual(get_base_url(), "http://freeflow.test:9999/v1")

    def test_api_key_from_environment(self):
        with patch.dict(os.environ, {"FREEFLOW_TOKEN": "secret"}, clear=True):
            self.assertEqual(get_api_key(), "secret")
        with patch.dict(os.environ, {"G4F_FREEFLOW_TOKEN": "other"}, clear=True):
            self.assertEqual(get_api_key(), "other")

    def test_api_key_from_daemon_token_file(self):
        with tempfile.TemporaryDirectory() as directory:
            token_file = Path(directory) / ".pi" / "agent" / "token"
            token_file.parent.mkdir(parents=True)
            token_file.write_text(" file-secret \n", encoding="utf-8")
            with patch.dict(
                os.environ, {"HOME": directory, "PI_FREEFLOW_TOKEN_FILE": str(token_file)},
                clear=True,
            ):
                self.assertEqual(get_token_file(), token_file)
                self.assertEqual(get_api_key(), "file-secret")

    def test_missing_token_file_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"HOME": directory}, clear=True
        ):
            self.assertIsNone(get_api_key())


class TestPiFreeflowModels(unittest.TestCase):
    def setUp(self):
        PiFreeflow.models = []
        PiFreeflow.default_model = None
        PiFreeflow.models_loaded = False
        self.addCleanup(self._reset)

    def _reset(self):
        PiFreeflow.models = []
        PiFreeflow.default_model = None
        PiFreeflow.models_loaded = False

    def test_get_models_uses_local_endpoint_and_token(self):
        module = importlib.import_module("g4f.Provider.template.OpenaiTemplate")
        response = FakeResponse(
            {"data": [{"id": "free-model"}, {"id": "other-model", "vision": True}]}
        )
        with patch.dict(
            os.environ,
            {
                "PI_FREEFLOW_BASE_URL": "http://freeflow.test/v1",
                "FREEFLOW_TOKEN": "secret",
            },
            clear=True,
        ), patch.object(module.requests, "get", return_value=response) as get:
            models = PiFreeflow.get_models()

        self.assertEqual(models, ["free-model", "other-model"])
        self.assertEqual(PiFreeflow.default_model, "free-model")
        self.assertEqual(PiFreeflow.vision_models, ["other-model"])
        self.assertEqual(get.call_args.args[0], "http://freeflow.test/v1/models")
        self.assertEqual(
            get.call_args.kwargs["headers"]["Authorization"], "Bearer secret"
        )

    def test_get_models_is_cached(self):
        module = importlib.import_module("g4f.Provider.template.OpenaiTemplate")
        response = FakeResponse({"data": [{"id": "free-model"}]})
        with patch.dict(
            os.environ, {"FREEFLOW_TOKEN": "secret"}, clear=True
        ), patch.object(module.requests, "get", return_value=response) as get:
            PiFreeflow.get_models()
            PiFreeflow.get_models()

        self.assertEqual(get.call_count, 1)

    def test_get_models_returns_empty_when_proxy_is_down(self):
        module = importlib.import_module("g4f.Provider.template.OpenaiTemplate")
        with patch.dict(os.environ, {}, clear=True), patch.object(
            module.requests, "get", side_effect=OSError("connection refused")
        ):
            self.assertEqual(PiFreeflow.get_models(), [])
        self.assertIsNone(PiFreeflow.default_model)


class TestPiFreeflowProviderRegistration(unittest.IsolatedAsyncioTestCase):
    def test_provider_loader_finds_the_provider(self):
        self.assertIs(ProviderLoader.from_name("PiFreeflow"), PiFreeflow)
        self.assertIs(ProviderLoader.from_name("pi-freeflow"), PiFreeflow)
        self.assertIs(ProviderLoader.from_name("Freeflow"), PiFreeflow)
        self.assertIs(Freeflow, PiFreeflow)

    async def test_chat_injects_token_and_base_url(self):
        captured = {}

        async def fake_create(cls, model, messages, **kwargs):
            captured.update(kwargs)
            yield "ok"

        with patch.dict(
            os.environ,
            {
                "PI_FREEFLOW_BASE_URL": "http://freeflow.test/v1",
                "FREEFLOW_TOKEN": "secret",
            },
            clear=True,
        ), patch.object(
            OpenaiTemplate,
            "create_async_generator",
            new=classmethod(fake_create),
        ):
            chunks = [
                chunk
                async for chunk in PiFreeflow.create_async_generator(
                    "free-model", [{"role": "user", "content": "hi"}]
                )
            ]

        self.assertEqual(chunks, ["ok"])
        self.assertEqual(captured["api_key"], "secret")
        self.assertEqual(captured["base_url"], "http://freeflow.test/v1")


if __name__ == "__main__":
    unittest.main()
