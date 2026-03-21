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

    @patch('g4f.gui.server.backend_api.socket.getaddrinfo')
    def test_is_safe_url_with_backslash_confusion(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 0))]
        from g4f.gui.server.backend_api import _is_safe_url
        self.assertFalse(_is_safe_url('http://127.0.0.1:6666\\@www.baidu.com'))

    @patch('g4f.gui.server.backend_api.socket.getaddrinfo')
    def test_is_safe_url_blocks_private(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 0))]
        from g4f.gui.server.backend_api import _is_safe_url
        self.assertFalse(_is_safe_url('http://127.0.0.1'))

    @patch('g4f.gui.server.backend_api.socket.getaddrinfo')
    def test_is_safe_url_allows_public(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 0))]
        from g4f.gui.server.backend_api import _is_safe_url
        self.assertTrue(_is_safe_url('http://example.com'))

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
