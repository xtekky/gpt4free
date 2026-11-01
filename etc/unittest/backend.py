from __future__ import annotations

import unittest
import asyncio
import socket
from unittest.mock import MagicMock, patch
from g4f.errors import MissingRequirementsError

try:
    from g4f.gui.server.backend_api import Backend_Api

    has_requirements = True
except Exception:
    has_requirements = False
try:
    from g4f.tools.web_search import search

    has_search = True
except Exception:
    has_search = False
try:
    from ddgs.exceptions import DDGSException
except ImportError:

    class DDGSException(Exception):
        pass


class TestProviderLoaderCache(unittest.TestCase):
    """The provider list is imported once and reused across GUI requests."""

    def setUp(self):
        from g4f.Provider import ProviderLoader

        self.loader = ProviderLoader
        self._saved_cache = ProviderLoader._providers_cache
        self._saved_key = ProviderLoader._providers_cache_key
        self._saved_ignored = list(ProviderLoader.ignored)
        ProviderLoader._providers_cache = None
        ProviderLoader._providers_cache_key = None
        ProviderLoader.ignored = []

    def tearDown(self):
        ProviderLoader = self.loader
        ProviderLoader._providers_cache = self._saved_cache
        ProviderLoader._providers_cache_key = self._saved_key
        ProviderLoader.ignored = self._saved_ignored

    def test_all_caches_provider_list(self):
        with patch.object(
            self.loader, "from_name", side_effect=lambda name: name
        ) as from_name:
            first = self.loader.all()
            second = self.loader.all()
            expected = len(self.loader.names)
            self.assertEqual(from_name.call_count, expected)
            self.assertIs(first, second)

    def test_all_invalidates_when_ignored_changes(self):
        with patch.object(
            self.loader, "from_name", side_effect=lambda name: name
        ) as from_name:
            self.loader.all()
            first_count = from_name.call_count
            self.loader.ignored = [self.loader.names[0]]
            providers = self.loader.all()
            self.assertEqual(from_name.call_count, first_count * 2 - 1)
            self.assertNotIn(self.loader.names[0], providers)


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

    def test_get_providers(self):
        response = self.api.get_providers()
        self.assertIsInstance(response, list)
        self.assertTrue(len(response) > 0)

    @patch("g4f.image.socket.getaddrinfo")
    def test_is_safe_url_with_backslash_confusion(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))
        ]
        from g4f.gui.server.backend_api import _is_safe_url

        self.assertFalse(_is_safe_url("http://127.0.0.1:6666\\@www.baidu.com"))

    @patch("g4f.image.socket.getaddrinfo")
    def test_is_safe_url_blocks_private(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))
        ]
        from g4f.gui.server.backend_api import _is_safe_url

        self.assertFalse(_is_safe_url("http://127.0.0.1"))

    @patch("g4f.image.socket.getaddrinfo")
    def test_is_safe_url_allows_public(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))
        ]
        from g4f.gui.server.backend_api import _is_safe_url

        self.assertTrue(_is_safe_url("http://example.com"))

    def test_search(self):
        if not has_search:
            self.skipTest("import error")
            return
        try:
            result = asyncio.run(search("Hello"))
        except DDGSException as e:
            self.skipTest(e)
        except MissingRequirementsError:
            self.skipTest("search is not installed")
        self.assertGreater(len(result), 0)
