"""Tests for the Google search provider's result handling.

These tests mock the CDP session so they run without a browser. They verify
that the provider keeps polling until the DOM yields results and that it
fails loudly (instead of silently returning nothing) when no results appear.
"""

import importlib
import unittest
from unittest.mock import AsyncMock, patch

from g4f.providers.response import SearchResults

google_search_module = importlib.import_module("g4f.Provider.search.GoogleSearch")
GoogleSearch = google_search_module.GoogleSearch


class FakeCDPSession:
    instances = []

    def __init__(self):
        self.closed = False
        self.navigated_to = None
        FakeCDPSession.instances.append(self)

    async def start(self):
        pass

    async def navigate(self, url):
        self.navigated_to = url

    async def click_button_by_text(self):
        pass

    async def close(self):
        self.closed = True


class TestGoogleSearchResults(unittest.IsolatedAsyncioTestCase):
    async def test_retries_until_results_are_available(self):
        expected = [
            {
                "title": "Example",
                "link": "https://example.com/",
                "snippet": "An example result.",
            }
        ]
        calls = 0

        async def read_results(session):
            nonlocal calls
            calls += 1
            return [] if calls < 3 else expected

        FakeCDPSession.instances = []
        with (
            patch.object(google_search_module, "CDPSession", FakeCDPSession),
            patch.object(google_search_module.asyncio, "sleep", AsyncMock()),
            patch.object(GoogleSearch, "_read_search_results", read_results),
        ):
            collected = [
                item
                async for item in GoogleSearch.create_async_generator(
                    "search", [{"role": "user", "content": "hello"}]
                )
            ]

        search_results = [item for item in collected if isinstance(item, SearchResults)]
        self.assertEqual(calls, 3)
        self.assertEqual(len(search_results), 1)
        self.assertEqual(search_results[0].list, expected)
        self.assertTrue(FakeCDPSession.instances[-1].closed)

    async def test_raises_when_no_results_are_found(self):
        async def read_results(session):
            return []

        FakeCDPSession.instances = []
        with (
            patch.object(google_search_module, "CDPSession", FakeCDPSession),
            patch.object(google_search_module.asyncio, "sleep", AsyncMock()),
            patch.object(GoogleSearch, "_read_search_results", read_results),
        ):
            with self.assertRaises(RuntimeError):
                async for _ in GoogleSearch.create_async_generator(
                    "search", [{"role": "user", "content": "hello"}]
                ):
                    pass

        self.assertTrue(FakeCDPSession.instances[-1].closed)
