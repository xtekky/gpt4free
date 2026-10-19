from __future__ import annotations

import importlib
import unittest

import g4f.Provider as Provider
from g4f.client import AsyncClient, ChatCompletion, ChatCompletionChunk
from g4f.providers.retry_provider import IterListProvider, _resolve_model
from g4f.Provider import ProviderLoader
from .mocks import (
    YieldProviderMock,
    RaiseExceptionProviderMock,
    AsyncRaiseExceptionProviderMock,
    YieldNoneProviderMock,
)

DEFAULT_MESSAGES = [{"role": "user", "content": "Hello"}]


class TestResolveModel(unittest.TestCase):
    def test_keeps_default_model_for_empty_model(self):
        class ProviderMock:
            default_model = "default-model"
            model_aliases = {"alias": "alias-target"}

        self.assertEqual(_resolve_model(ProviderMock, ""), "default-model")
        self.assertEqual(_resolve_model(ProviderMock, "alias"), "alias-target")
        self.assertEqual(_resolve_model(ProviderMock, "unknown"), "unknown")


class TestProviderLoader(unittest.TestCase):
    def test_submodule_does_not_shadow_provider_class(self):
        # Importing g4f.Provider.ChatGPT sets the module as a package attribute;
        # accessing the provider must still return the provider class.
        importlib.import_module("g4f.Provider.ChatGPT")
        provider = getattr(Provider, "ChatGPT")
        self.assertTrue(getattr(provider, "working", False))
        self.assertIs(provider, ProviderLoader.from_name("ChatGPT"))


class TestIterListProvider(unittest.IsolatedAsyncioTestCase):
    async def test_skip_provider(self):
        client = AsyncClient(
            provider=IterListProvider(
                [RaiseExceptionProviderMock, YieldProviderMock], False
            )
        )
        response = await client.chat.completions.create(DEFAULT_MESSAGES, "")
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("Hello", response.choices[0].message.content)

    async def test_only_one_result(self):
        client = AsyncClient(
            provider=IterListProvider([YieldProviderMock, YieldProviderMock])
        )
        response = await client.chat.completions.create(DEFAULT_MESSAGES, "")
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("Hello", response.choices[0].message.content)

    async def test_stream_skip_provider(self):
        client = AsyncClient(
            provider=IterListProvider(
                [AsyncRaiseExceptionProviderMock, YieldProviderMock], False
            )
        )
        messages = [
            {"role": "user", "content": chunk} for chunk in ["How ", "are ", "you", "?"]
        ]
        response = client.chat.completions.create(messages, "Hello", stream=True)
        async for chunk in response:
            chunk: ChatCompletionChunk = chunk
            self.assertIsInstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content is not None:
                self.assertIsInstance(chunk.choices[0].delta.content, str)

    async def test_stream_only_one_result(self):
        client = AsyncClient(
            provider=IterListProvider([YieldProviderMock, YieldProviderMock], False)
        )
        messages = [{"role": "user", "content": chunk} for chunk in ["You ", "You "]]
        response = client.chat.completions.create(
            messages, "Hello", stream=True, max_tokens=2
        )
        response_list = []
        async for chunk in response:
            response_list.append(chunk)
        self.assertEqual(len(response_list), 3)
        for chunk in response_list:
            if chunk.choices[0].delta.content is not None:
                self.assertEqual(chunk.choices[0].delta.content, "You ")

    async def test_skip_none(self):
        client = AsyncClient(
            provider=IterListProvider([YieldNoneProviderMock, YieldProviderMock], False)
        )
        response = await client.chat.completions.create(DEFAULT_MESSAGES, "")
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("Hello", response.choices[0].message.content)

    async def test_stream_skip_none(self):
        client = AsyncClient(
            provider=IterListProvider([YieldNoneProviderMock, YieldProviderMock], False)
        )
        response = client.chat.completions.create(DEFAULT_MESSAGES, "", stream=True)
        response_list = [chunk async for chunk in response]
        self.assertEqual(len(response_list), 2)
        for chunk in response_list:
            if chunk.choices[0].delta.content is not None:
                self.assertEqual(chunk.choices[0].delta.content, "Hello")
