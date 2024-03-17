import unittest
import asyncio
from unittest.mock import MagicMock
from .mocks import ProviderMock
import g4f
from g4f.errors import MissingRequirementsError

try:
    from g4f.gui.server.backend import Backend_Api, get_error_message
    has_requirements = True
except:
    has_requirements = False

class TestBackendApi(unittest.TestCase):

    def setUp(self):
        if not has_requirements:
            self.skipTest("gui is not installed")
        self.app = MagicMock()
        self.api = Backend_Api(self.app)

    def test_version(self):
        response = self.api.get_version()
        self.assertIn("version", response)
        self.assertIn("latest_version", response)

    def test_get_models(self):
        response = self.api.get_models()
        self.assertIsInstance(response, list)
        self.assertTrue(len(response) > 0)

    def test_get_providers(self):
        response = self.api.get_providers()
        self.assertIsInstance(response, list)
        self.assertTrue(len(response) > 0)

    def test_search(self):
        from g4f.gui.server.internet import search
        try:
            result = asyncio.run(search("Hello"))
        except MissingRequirementsError:
            self.skipTest("search is not installed")
        self.assertEqual(5, len(result))

class TestUtilityFunctions(unittest.TestCase):

    def setUp(self):
        if not has_requirements:
            self.skipTest("gui is not installed")

    def test_get_error_message(self):
        g4f.debug.last_provider = ProviderMock
        exception = Exception("Message")
        result = get_error_message(exception)
        self.assertEqual("ProviderMock: Exception: Message", result)

if __name__ == '__main__':
    unittest.main()