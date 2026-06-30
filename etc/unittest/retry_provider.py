from __future__ import annotations

import unittest

from g4f.client import AsyncClient, ChatCompletion, ChatCompletionChunk
from g4f.providers.retry_provider import IterListProvider
from .mocks import YieldProviderMock, RaiseExceptionProviderMock, AsyncRaiseExceptionProviderMock, YieldNoneProviderMock

DEFAULT_MESSAGES = [{'role': 'user', 'content': 'Hello'}]

class TestIterListProvider(unittest.IsolatedAsyncioTestCase):

    async def test_skip_provider(self):
        client = AsyncClient(provider=IterListProvider([RaiseExceptionProviderMock, YieldProviderMock], False))
        response = await client.chat.completions.create(DEFAULT_MESSAGES, "")
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("Hello", response.choices[0].message.content)

    async def test_only_one_result(self):
        client = AsyncClient(provider=IterListProvider([YieldProviderMock, YieldProviderMock]))
        response = await client.chat.completions.create(DEFAULT_MESSAGES, "")
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("Hello", response.choices[0].message.content)

    async def test_stream_skip_provider(self):
        client = AsyncClient(provider=IterListProvider([AsyncRaiseExceptionProviderMock, YieldProviderMock], False))
        messages = [{'role': 'user', 'content': chunk} for chunk in ["How ", "are ", "you", "?"]]
        response = client.chat.completions.create(messages, "Hello", stream=True)
        async for chunk in response:
            chunk: ChatCompletionChunk = chunk
            self.assertIsInstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content is not None:
                self.assertIsInstance(chunk.choices[0].delta.content, str)
 
    async def test_stream_only_one_result(self):
        client = AsyncClient(provider=IterListProvider([YieldProviderMock, YieldProviderMock], False))
        messages = [{'role': 'user', 'content': chunk} for chunk in ["You ", "You "]]
        response = client.chat.completions.create(messages, "Hello", stream=True, max_tokens=2)
        response_list = []
        async for chunk in response:
            response_list.append(chunk)
        self.assertEqual(len(response_list), 3)
        for chunk in response_list:
            if chunk.choices[0].delta.content is not None:
                self.assertEqual(chunk.choices[0].delta.content, "You ")

    async def test_skip_none(self):
        client = AsyncClient(provider=IterListProvider([YieldNoneProviderMock, YieldProviderMock], False))
        response = await client.chat.completions.create(DEFAULT_MESSAGES, "")
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("Hello", response.choices[0].message.content)

    async def test_stream_skip_none(self):
        client = AsyncClient(provider=IterListProvider([YieldNoneProviderMock, YieldProviderMock], False))
        response = client.chat.completions.create(DEFAULT_MESSAGES, "", stream=True)
        response_list = [chunk async for chunk in response]
        self.assertEqual(len(response_list), 2)
        for chunk in response_list:
            if chunk.choices[0].delta.content is not None:
                self.assertEqual(chunk.choices[0].delta.content, "Hello")