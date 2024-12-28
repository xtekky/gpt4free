from __future__ import annotations

import json
import unittest

try:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import DuckDuckGoSearchException
    from bs4 import BeautifulSoup
    has_requirements = True
except ImportError:
    has_requirements = False

from g4f.client import AsyncClient
from .mocks import YieldProviderMock

DEFAULT_MESSAGES = [{'role': 'user', 'content': 'Hello'}]

class TestIterListProvider(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        if not has_requirements:
            self.skipTest('web search requirements not passed')

    async def test_search(self):
        client = AsyncClient(provider=YieldProviderMock)
        tool_calls = [
            {
                "function": {
                    "arguments": {
                        "query": "search query", # content of last message: messages[-1]["content"]
                        "max_results": 5, # maximum number of search results
                        "max_words": 500, # maximum number of used words from search results for generating the response
                        "backend": "html", # or "lite", "api": change it to pypass rate limits
                        "add_text": True, # do scraping websites
                        "timeout": 5, # in seconds for scraping websites
                        "region": "wt-wt",
                        "instructions": "Using the provided web search results, to write a comprehensive reply to the user request.\n"
                                        "Make sure to add the sources of cites using [[Number]](Url) notation after the reference. Example: [[0]](http://google.com)",
                    },
                    "name": "search_tool"
                },
                "type": "function"
            }
        ]
        try:
            response = await client.chat.completions.create([{"content": "", "role": "user"}], "", tool_calls=tool_calls)
            self.assertIn("Using the provided web search results", response.choices[0].message.content)
        except DuckDuckGoSearchException as e:
            self.skipTest(f'DuckDuckGoSearchException: {e}')

    async def test_search2(self):
        client = AsyncClient(provider=YieldProviderMock)
        tool_calls = [
            {
                "function": {
                    "arguments": {
                        "query": "search query",
                    },
                    "name": "search_tool"
                },
                "type": "function"
            }
        ]
        try:
            response = await client.chat.completions.create([{"content": "", "role": "user"}], "", tool_calls=tool_calls)
            self.assertIn("Using the provided web search results", response.choices[0].message.content)
        except DuckDuckGoSearchException as e:
            self.skipTest(f'DuckDuckGoSearchException: {e}')

    async def test_search3(self):
        client = AsyncClient(provider=YieldProviderMock)
        tool_calls = [
            {
                "function": {
                    "arguments": json.dumps({
                        "query": "search query", # content of last message: messages[-1]["content"]
                        "max_results": 5, # maximum number of search results
                        "max_words": 500, # maximum number of used words from search results for generating the response
                    }),
                    "name": "search_tool"
                },
                "type": "function"
            }
        ]
        try:
            response = await client.chat.completions.create([{"content": "", "role": "user"}], "", tool_calls=tool_calls)
            self.assertIn("Using the provided web search results", response.choices[0].message.content)
        except DuckDuckGoSearchException as e:
            self.skipTest(f'DuckDuckGoSearchException: {e}')