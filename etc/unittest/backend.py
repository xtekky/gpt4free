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
        """Testing the search functionality"""
        from g4f.gui.server.internet import search, SearchResults
        try:
            # Perform a search with a specific query
            query = "hello"
            result = asyncio.run(search(query, n_results=3, max_words=250))
            
            # Check that we have received the SearchResults object
            self.assertIsInstance(result, SearchResults)
            
            # Check the availability of results
            self.assertTrue(len(result.results) > 0, "Search should return at least one result")
            
            # Check the structure of the results
            for entry in result.results:
                # Check for all the necessary fields
                self.assertTrue(hasattr(entry, 'title'), "Result should have title")
                self.assertTrue(hasattr(entry, 'url'), "Result should have URL")
                self.assertTrue(hasattr(entry, 'snippet'), "Result should have snippet")
                
                # Check that the fields are not empty
                self.assertIsNotNone(entry.title)
                self.assertIsNotNone(entry.url)
                self.assertTrue(entry.snippet or entry.text, "Result should have either snippet or text")
                
                # Check the URL format
                self.assertTrue(
                    entry.url.startswith(('http://', 'https://')), 
                    f"Invalid URL format: {entry.url}"
                )
                
            # Check the word count
            self.assertIsInstance(result.used_words, int)
            self.assertGreaterEqual(result.used_words, 0)
            
            # Check the word countaCheck that we can get a string representation of the results
            str_result = str(result)
            self.assertIsInstance(str_result, str)
            self.assertGreater(len(str_result), 0)

        except DuckDuckGoSearchException as e:
            self.skipTest(f"DuckDuckGo search failed: {str(e)}")
        except MissingRequirementsError:
            self.skipTest("Search requirements not installed")
        except Exception as e:
            self.fail(f"Unexpected error during search: {str(e)}")

    def tearDown(self):
        # Closing the event loop after tests
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.close()

if __name__ == '__main__':
    unittest.main()
