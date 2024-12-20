from __future__ import annotations

import unittest
import asyncio
from unittest.mock import MagicMock
from g4f.errors import MissingRequirementsError

# Check for required modules
try:
    from g4f.gui.server.backend import Backend_Api
    has_requirements = True
except ImportError:
    has_requirements = False

try:
    from duckduckgo_search.exceptions import DuckDuckGoSearchException
except ImportError:
    # Create stub for exception if module not installed
    class DuckDuckGoSearchException(Exception):
        pass

try:
    from duckduckgo_search import AsyncDDGS
    has_search = True
except ImportError:
    has_search = False

class TestBackendApi(unittest.TestCase):
    """Tests for Backend API."""

    def setUp(self):
        """Set up test environment."""
        if not has_requirements:
            self.skipTest("GUI requirements are not installed")
        self.app = MagicMock()
        self.api = Backend_Api(self.app)

    def test_version(self):
        """Test getting version."""
        response = self.api.get_version()
        self.assertIn("version", response)
        self.assertIn("latest_version", response)

    def test_get_models(self):
        """Test getting list of models."""
        response = self.api.get_models()
        self.assertIsInstance(response, list)
        self.assertTrue(len(response) > 0)

    def test_get_providers(self):
        """Test getting list of providers."""
        response = self.api.get_providers()
        self.assertIsInstance(response, list)
        self.assertTrue(len(response) > 0)

    def test_search(self):
        """Test search function."""
        if not has_search:
            self.skipTest("DuckDuckGo search is not installed")

        from g4f.gui.server.internet import search, SearchResults
        try:
            # Perform search query
            result = asyncio.run(search("Hello"))
            
            # Check if result is SearchResults instance
            self.assertIsInstance(result, SearchResults)
            
            # Check if results exist
            self.assertTrue(len(result.results) >= 4)
            
            # Verify result structure
            if result.results:
                first_result = result.results[0]
                self.assertTrue(hasattr(first_result, 'title'))
                self.assertTrue(hasattr(first_result, 'url'))
                self.assertTrue(hasattr(first_result, 'snippet'))
                
        except DuckDuckGoSearchException as e:
            self.skipTest(f"DuckDuckGo search error: {str(e)}")
        except MissingRequirementsError:
            self.skipTest("Search requirements are not installed")
        except Exception as e:
            self.fail(f"Unexpected error during search: {str(e)}")

if __name__ == '__main__':
    unittest.main()
