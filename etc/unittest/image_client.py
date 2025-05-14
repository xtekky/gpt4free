from __future__ import annotations

import asyncio
import unittest

from g4f.client import AsyncClient, ImagesResponse
from g4f.providers.retry_provider import IterListProvider
from .mocks import (
    YieldImageResponseProviderMock,
    MissingAuthProviderMock,
    AsyncRaiseExceptionProviderMock,
    YieldNoneProviderMock
)

DEFAULT_MESSAGES = [{'role': 'user', 'content': 'Hello'}]

class TestIterListProvider(unittest.IsolatedAsyncioTestCase):

    async def test_skip_provider(self):
        client = AsyncClient(image_provider=IterListProvider([MissingAuthProviderMock, YieldImageResponseProviderMock], False))
        response = await client.images.generate("Hello", "", response_format="orginal")
        self.assertIsInstance(response, ImagesResponse)
        self.assertEqual("Hello", response.data[0].url)

    async def test_only_one_result(self):
        client = AsyncClient(image_provider=IterListProvider([YieldImageResponseProviderMock, YieldImageResponseProviderMock], False))
        response = await client.images.generate("Hello", "", response_format="orginal")
        self.assertIsInstance(response, ImagesResponse)
        self.assertEqual("Hello", response.data[0].url)

    async def test_skip_none(self):
        client = AsyncClient(image_provider=IterListProvider([YieldNoneProviderMock, YieldImageResponseProviderMock], False))
        response = await client.images.generate("Hello", "", response_format="orginal")
        self.assertIsInstance(response, ImagesResponse)
        self.assertEqual("Hello", response.data[0].url)

    def test_raise_exception(self):
        async def run_exception():
            client = AsyncClient(image_provider=IterListProvider([YieldNoneProviderMock, AsyncRaiseExceptionProviderMock], False))
            await client.images.generate("Hello", "")
        self.assertRaises(RuntimeError, asyncio.run, run_exception())

if __name__ == '__main__':
    unittest.main()