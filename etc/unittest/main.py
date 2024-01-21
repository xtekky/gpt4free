from .include import DEFAULT_MESSAGES
import unittest
import asyncio
import g4f
from g4f import ChatCompletion, get_last_provider
from g4f.Provider import RetryProvider
from .mocks import ProviderMock

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

if __name__ == '__main__':
    unittest.main()