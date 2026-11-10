from __future__ import annotations

import importlib
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from g4f.Provider import ProviderLoader, UberLLM
from g4f.client import Client
from g4f.errors import MissingAuthError
from g4f.tools.auth import AuthManager


OPENAI_TEMPLATE_MODULE = importlib.import_module("g4f.Provider.template.OpenaiTemplate")


class TestUberLLM(unittest.TestCase):
    def setUp(self):
        self.models_patch = patch.object(UberLLM, "models", {})
        self.models_patch.start()
        self.addCleanup(self.models_patch.stop)

    def test_named_provider_and_environment_key(self):
        self.assertIs(ProviderLoader.from_name("UberLLM"), UberLLM)
        self.assertIn("UberLLM", ProviderLoader.extra)
        with patch.dict(os.environ, {"UBERLLM_API_KEY": "ull-test-value"}, clear=True):
            self.assertEqual(AuthManager.load_api_key(UberLLM), "ull-test-value")

    @patch.object(OPENAI_TEMPLATE_MODULE.requests, "get")
    def test_model_discovery_works_without_a_key(self, get):
        get.return_value.json.return_value = {
            "data": [{"id": "deepseek/deepseek-v4.1-flash"}, {"id": "anthropic/claude-sonnet-4.6"}]
        }
        with patch.dict(os.environ, {}, clear=True):
            models = UberLLM.get_models(timeout=10)
        self.assertEqual(list(models), ["deepseek/deepseek-v4.1-flash", "anthropic/claude-sonnet-4.6"])
        self.assertEqual(get.call_args.args[0], "https://api.uberllm.dev/v1/models")
        self.assertNotIn("Authorization", get.call_args.kwargs["headers"])

    def test_chat_requires_a_key_without_network_io(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            OPENAI_TEMPLATE_MODULE, "StreamSession"
        ) as session_type:
            client = Client(provider=UberLLM)
            with self.assertRaises(MissingAuthError):
                client.chat.completions.create(
                    model="deepseek/deepseek-v4.1-flash",
                    messages=[{"role": "user", "content": "Hi"}],
                )
            session_type.assert_not_called()

    @patch.object(OPENAI_TEMPLATE_MODULE, "StreamSession")
    @patch.object(OPENAI_TEMPLATE_MODULE.requests, "get")
    def test_client_keeps_vendor_prefixed_model_ids(self, get, session_type):
        get.return_value.json.return_value = {"data": [{"id": "anthropic/claude-sonnet-4.6"}]}
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        response = MagicMock(status=200, headers={"content-type": "application/json"})
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        response.json = AsyncMock(return_value={
            "model": "anthropic/claude-sonnet-4.6",
            "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
        })
        session.post.return_value = response
        session_type.return_value = session

        client = Client(provider=UberLLM, api_key="ull-test-value")
        result = client.chat.completions.create(
            model="anthropic/claude-sonnet-4.6", messages=[{"role": "user", "content": "Reply OK"}], stream=False,
        )
        self.assertEqual(result.choices[0].message.content, "OK")
        self.assertEqual(session.post.call_args.args[0], "https://api.uberllm.dev/v1/chat/completions")
        self.assertEqual(session.post.call_args.kwargs["json"]["model"], "anthropic/claude-sonnet-4.6")
        self.assertEqual(session_type.call_args.kwargs["headers"]["Authorization"], "Bearer ull-test-value")
