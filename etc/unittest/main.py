import unittest
import asyncio
import g4f
from g4f import ChatCompletion, get_last_provider
from g4f.Provider import RetryProvider
from .mocks import ProviderMock

DEFAULT_MESSAGES = [{'role': 'user', 'content': 'Hello'}]

class NoTestChatCompletion(unittest.TestCase):

    def no_test_create_default(self):
        result = ChatCompletion.create(g4f.models.default, DEFAULT_MESSAGES)
        if "Good" not in result and "Hi" not in result:
            self.assertIn("Hello", result)

    def no_test_bing_provider(self):
        provider = g4f.Provider.Bing
        result = ChatCompletion.create(g4f.models.default, DEFAULT_MESSAGES, provider)
        self.assertIn("Bing", result)

class TestGetLastProvider(unittest.TestCase):

    def test_get_last_provider(self):
        ChatCompletion.create(g4f.models.default, DEFAULT_MESSAGES, ProviderMock)
        self.assertEqual(get_last_provider(), ProviderMock)

    def test_get_last_provider_retry(self):
        ChatCompletion.create(g4f.models.default, DEFAULT_MESSAGES, RetryProvider([ProviderMock]))
        self.assertEqual(get_last_provider(), ProviderMock)

    def test_get_last_provider_async(self):
        coroutine = ChatCompletion.create_async(g4f.models.default, DEFAULT_MESSAGES, ProviderMock)
        asyncio.run(coroutine)
        self.assertEqual(get_last_provider(), ProviderMock)

    def test_get_last_provider_as_dict(self):
        ChatCompletion.create(g4f.models.default, DEFAULT_MESSAGES, ProviderMock)
        last_provider_dict = get_last_provider(True)
        self.assertIsInstance(last_provider_dict, dict)
        self.assertIn('name', last_provider_dict)
        self.assertEqual(ProviderMock.__name__, last_provider_dict['name'])