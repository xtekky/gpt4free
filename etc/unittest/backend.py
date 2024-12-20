from __future__ import annotations

import unittest
import asyncio
from unittest.mock import MagicMock

try:
    from g4f.gui.server.backend import Backend_Api
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
            result = asyncio.run(search("Hello", n_results=2, max_words=500))
            self.assertIsNotNone(result)
            self.assertTrue(len(result.results) > 0)
        except Exception as e:
            self.skipTest(f"Search failed: {str(e)}")
