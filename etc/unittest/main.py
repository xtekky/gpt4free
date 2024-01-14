import sys
import pathlib
import unittest
from unittest.mock import MagicMock

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import g4f
from g4f import ChatCompletion, get_last_provider
from g4f.gui.server.backend import Backend_Api, get_error_message
from g4f.base_provider import BaseProvider

g4f.debug.logging = False
g4f.debug.version_check = False

class MockProvider(BaseProvider):
    working = True

    def create_completion(
        model, messages, stream, **kwargs
    ):
        yield "Mock"

    async def create_async(
        model, messages, **kwargs
    ):
        return "Mock"

class TestBackendApi(unittest.TestCase):

    def setUp(self):
        self.app = MagicMock()
        self.api = Backend_Api(self.app)

    def test_version(self):
        response = self.api.get_version()
        self.assertIn("version", response)
        self.assertIn("latest_version", response)

class TestChatCompletion(unittest.TestCase):

    def test_create(self):
        messages = [{'role': 'user', 'content': 'Hello'}]
        result = ChatCompletion.create(g4f.models.default, messages)
        self.assertTrue("Hello" in result or "Good" in result)
        
    def test_get_last_provider(self):
        messages = [{'role': 'user', 'content': 'Hello'}]
        ChatCompletion.create(g4f.models.default, messages, MockProvider)
        self.assertEqual(get_last_provider(), MockProvider)
        
    def test_bing_provider(self):
        messages = [{'role': 'user', 'content': 'Hello'}]
        provider = g4f.Provider.Bing
        result = ChatCompletion.create(g4f.models.default, messages, provider)
        self.assertTrue("Bing" in result)

class TestChatCompletionAsync(unittest.IsolatedAsyncioTestCase):
    
    async def test_async(self):
        messages = [{'role': 'user', 'content': 'Hello'}]
        result = await ChatCompletion.create_async(g4f.models.default, messages, MockProvider)
        self.assertTrue("Mock" in result)

class TestUtilityFunctions(unittest.TestCase):

    def test_get_error_message(self):
        g4f.debug.last_provider = g4f.Provider.Bing
        exception = Exception("Message")
        result = get_error_message(exception)
        self.assertEqual("Bing: Exception: Message", result)

if __name__ == '__main__':
    unittest.main()