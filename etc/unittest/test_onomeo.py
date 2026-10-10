from __future__ import annotations

import importlib
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from g4f.Provider import Onomeo, ProviderLoader
from g4f.client import Client
from g4f.errors import MissingAuthError
from g4f.tools.auth import AuthManager


OPENAI_TEMPLATE_MODULE = importlib.import_module("g4f.Provider.template.OpenaiTemplate")


class TestOnomeo(unittest.TestCase):
    def setUp(self):
        self.models_patch = patch.object(Onomeo, "models", {})
        self.models_patch.start()
        self.addCleanup(self.models_patch.stop)

    def test_named_provider_and_environment_key(self):
        self.assertIs(ProviderLoader.from_name("Onomeo"), Onomeo)
        self.assertIn("Onomeo", ProviderLoader.extra)
        with patch.dict(os.environ, {"ONOMEO_API_KEY": "sk-test-value"}, clear=True):
            self.assertEqual(AuthManager.load_api_key(Onomeo), "sk-test-value")

    def test_model_discovery_requires_a_key_without_network_io(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            OPENAI_TEMPLATE_MODULE.requests, "get"
        ) as get:
            with self.assertRaises(MissingAuthError):
                Onomeo.get_models()
            get.assert_not_called()

    @patch.object(OPENAI_TEMPLATE_MODULE.requests, "get")
    def test_authenticated_model_discovery_keeps_gateway_ids(self, get):
        get.return_value.json.return_value = {
            "data": [{"id": "auto"}, {"id": "deepseek-v4-flash"}]
        }
        models = Onomeo.get_models(api_key="sk-test-value", timeout=10)
        self.assertEqual(list(models), ["auto", "deepseek-v4-flash"])
        self.assertEqual(get.call_args.args[0], "https://onomeo.com/v1/models")
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], "Bearer sk-test-value")
        self.assertEqual(get.call_args.kwargs["timeout"], 10)

    @patch.object(OPENAI_TEMPLATE_MODULE, "StreamSession")
    @patch.object(OPENAI_TEMPLATE_MODULE.requests, "get")
    def test_client_sends_a_free_model_through_chat_completions(self, get, session_type):
        get.return_value.json.return_value = {"data": [{"id": "deepseek-v4-flash"}]}
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        response = MagicMock(status=200, headers={"content-type": "application/json"})
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        response.json = AsyncMock(return_value={
            "model": "deepseek-v4-flash",
            "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
        })
        session.post.return_value = response
        session_type.return_value = session

        client = Client(provider=Onomeo, api_key="sk-test-value")
        result = client.chat.completions.create(
            model="deepseek-v4-flash", messages=[{"role": "user", "content": "Reply OK"}], stream=False,
        )
        self.assertEqual(result.choices[0].message.content, "OK")
        self.assertEqual(session.post.call_args.args[0], "https://onomeo.com/v1/chat/completions")
        self.assertEqual(session.post.call_args.kwargs["json"]["model"], "deepseek-v4-flash")
        self.assertEqual(session_type.call_args.kwargs["headers"]["Authorization"], "Bearer sk-test-value")
