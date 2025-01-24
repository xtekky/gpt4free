from __future__ import annotations

import unittest

from g4f.errors import ModelNotFoundError
from g4f.client import Client, AsyncClient, ChatCompletion, ChatCompletionChunk, get_model_and_provider
from g4f.Provider.Copilot import Copilot
from g4f.models import gpt_4o
from .mocks import AsyncGeneratorProviderMock, ModelProviderMock, YieldProviderMock

DEFAULT_MESSAGES = [{'role': 'user', 'content': 'Hello'}]

class AsyncTestPassModel(unittest.IsolatedAsyncioTestCase):

    async def test_response(self):
        client = AsyncClient(provider=AsyncGeneratorProviderMock)
        response = await client.chat.completions.create(DEFAULT_MESSAGES, "")
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("Mock", response.choices[0].message.content)

    async def test_pass_model(self):
        client = AsyncClient(provider=ModelProviderMock)
        response = await client.chat.completions.create(DEFAULT_MESSAGES, "Hello")
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("Hello", response.choices[0].message.content)

    async def test_max_tokens(self):
        client = AsyncClient(provider=YieldProviderMock)
        messages = [{'role': 'user', 'content': chunk} for chunk in ["How ", "are ", "you", "?"]]
        response = await client.chat.completions.create(messages, "Hello", max_tokens=1)
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("How ", response.choices[0].message.content)
        response = await client.chat.completions.create(messages, "Hello", max_tokens=2)
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("How are ", response.choices[0].message.content)

    async def test_max_stream(self):
        client = AsyncClient(provider=YieldProviderMock)
        messages = [{'role': 'user', 'content': chunk} for chunk in ["How ", "are ", "you", "?"]]
        response = client.chat.completions.create(messages, "Hello", stream=True)
        async for chunk in response:
            chunk: ChatCompletionChunk = chunk
            self.assertIsInstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content is not None:
                self.assertIsInstance(chunk.choices[0].delta.content, str)
        messages = [{'role': 'user', 'content': chunk} for chunk in ["You ", "You ", "Other", "?"]]
        response = client.chat.completions.create(messages, "Hello", stream=True, max_tokens=2)
        response_list = []
        async for chunk in response:
            response_list.append(chunk)
        self.assertEqual(len(response_list), 3)
        for chunk in response_list:
            if chunk.choices[0].delta.content is not None:
                self.assertEqual(chunk.choices[0].delta.content, "You ")

    async def test_stop(self):
        client = AsyncClient(provider=YieldProviderMock)
        messages = [{'role': 'user', 'content': chunk} for chunk in ["How ", "are ", "you", "?"]]
        response = await client.chat.completions.create(messages, "Hello", stop=["and"])
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("How are you?", response.choices[0].message.content)

class TestPassModel(unittest.TestCase):

    def test_response(self):
        client = Client(provider=AsyncGeneratorProviderMock)
        response = client.chat.completions.create(DEFAULT_MESSAGES, "")
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("Mock", response.choices[0].message.content)

    def test_pass_model(self):
        client = Client(provider=ModelProviderMock)
        response = client.chat.completions.create(DEFAULT_MESSAGES, "Hello")
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("Hello", response.choices[0].message.content)

    def test_max_tokens(self):
        client = Client(provider=YieldProviderMock)
        messages = [{'role': 'user', 'content': chunk} for chunk in ["How ", "are ", "you", "?"]]
        response = client.chat.completions.create(messages, "Hello", max_tokens=1)
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("How ", response.choices[0].message.content)
        response = client.chat.completions.create(messages, "Hello", max_tokens=2)
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("How are ", response.choices[0].message.content)

    def test_max_stream(self):
        client = Client(provider=YieldProviderMock)
        messages = [{'role': 'user', 'content': chunk} for chunk in ["How ", "are ", "you", "?"]]
        response = client.chat.completions.create(messages, "Hello", stream=True)
        for chunk in response:
            self.assertIsInstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content is not None:
                self.assertIsInstance(chunk.choices[0].delta.content, str)
        messages = [{'role': 'user', 'content': chunk} for chunk in ["You ", "You ", "Other", "?"]]
        response = client.chat.completions.create(messages, "Hello", stream=True, max_tokens=2)
        response_list = list(response)
        self.assertEqual(len(response_list), 3)
        for chunk in response_list:
            if chunk.choices[0].delta.content is not None:
                self.assertEqual(chunk.choices[0].delta.content, "You ")

    def test_stop(self):
        client = Client(provider=YieldProviderMock)
        messages = [{'role': 'user', 'content': chunk} for chunk in ["How ", "are ", "you", "?"]]
        response = client.chat.completions.create(messages, "Hello", stop=["and"])
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("How are you?", response.choices[0].message.content)

    def test_model_not_found(self):
        def run_exception():
            client = Client()
            client.chat.completions.create(DEFAULT_MESSAGES, "Hello")
        self.assertRaises(ModelNotFoundError, run_exception)

    def test_best_provider(self):
        not_default_model = "gpt-4o"
        model, provider = get_model_and_provider(not_default_model, None, False)
        self.assertTrue(hasattr(provider, "create_completion"))
        self.assertEqual(model, not_default_model)

    def test_default_model(self):
        default_model = ""
        model, provider = get_model_and_provider(default_model, None, False)
        self.assertTrue(hasattr(provider, "create_completion"))
        self.assertEqual(model, default_model)

    def test_provider_as_model(self):
        provider_as_model = Copilot.__name__
        model, provider = get_model_and_provider(provider_as_model, None, False)
        self.assertTrue(hasattr(provider, "create_completion"))
        self.assertIsInstance(model, str)
        self.assertEqual(model, Copilot.default_model)

    def test_get_model(self):
        model, provider = get_model_and_provider(gpt_4o.name, None, False)
        self.assertTrue(hasattr(provider, "create_completion"))
        self.assertEqual(model, gpt_4o.name)

if __name__ == '__main__':
    unittest.main()
