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

    async def async_search(self, query: str) -> list:
        """Helper method to perform async search."""
        async with AsyncDDGS() as ddgs:
            results = await ddgs.atext(
                keywords=query,
                max_results=5,
                backend='api'
            )
            return results or []

    def test_search(self):
        """Test search function."""
        if not has_search:
            self.skipTest("DuckDuckGo search is not installed")

        try:
            # Direct test of DuckDuckGo search
            results = asyncio.run(self.async_search("Python programming"))
            
            # Basic validation of results
            self.assertIsInstance(results, list)
            
            # Log the results for debugging
            print(f"Search results count: {len(results)}")
            if results:
                print(f"First result: {results[0]}")
            
            # Verify at least some results are returned
            self.assertTrue(results, "No results returned from search")
            
            # Verify structure of results
            if results:
                first_result = results[0]
                self.assertIn('title', first_result)
                self.assertIn('link', first_result)
                self.assertTrue(
                    'body' in first_result or 'snippet' in first_result,
                    "Result should contain either 'body' or 'snippet'"
                )
                
        except DuckDuckGoSearchException as e:
            self.skipTest(f"DuckDuckGo search error: {str(e)}")
        except MissingRequirementsError:
            self.skipTest("Search requirements are not installed")
        except Exception as e:
            self.fail(f"Unexpected error during search: {str(e)}")

if __name__ == '__main__':
    unittest.main()
