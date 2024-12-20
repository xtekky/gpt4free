from __future__ import annotations

import unittest
import asyncio
from unittest.mock import MagicMock
from g4f.errors import MissingRequirementsError

try:
    from g4f.gui.server.backend import Backend_Api
    has_requirements = True
except:
    has_requirements = False

try:
    from duckduckgo_search.exceptions import DuckDuckGoSearchException
except ImportError:
    class DuckDuckGoSearchException:
        pass


class TestBackendApi(unittest.TestCase):
    """Test class for Backend API functionality"""

    def setUp(self):
        """Set up test environment"""
        if not has_requirements:
            self.skipTest("gui is not installed")
        self.app = MagicMock()
        self.api = Backend_Api(self.app)

    def test_version(self):
        """Test version endpoint"""
        response = self.api.get_version()
        self.assertIn("version", response)
        self.assertIn("latest_version", response)

    def test_get_models(self):
        """Test models retrieval"""
        response = self.api.get_models()
        self.assertIsInstance(response, list)
        self.assertTrue(len(response) > 0)

    def test_get_providers(self):
        """Test providers retrieval"""
        response = self.api.get_providers()
        self.assertIsInstance(response, list)
        self.assertTrue(len(response) > 0)

    def test_search(self):
        """Test search functionality"""
        from g4f.gui.server.internet import search
        try:
            result = asyncio.run(search(
                "Hello",
                n_results=2,        # Minimum required results
                add_text=False      # For faster execution
            ))
        except DuckDuckGoSearchException as e:
            self.skipTest(str(e))
        except MissingRequirementsError:
            self.skipTest("search is not installed")
        
        # Check if we have at least some results
        self.assertTrue(len(result.results) > 0)


if __name__ == '__main__':
    unittest.main()
