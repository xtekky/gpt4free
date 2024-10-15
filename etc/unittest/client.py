import unittest

from g4f.client import Client, ChatCompletion, ChatCompletionChunk
from .mocks import AsyncGeneratorProviderMock, ModelProviderMock, YieldProviderMock

DEFAULT_MESSAGES = [{'role': 'user', 'content': 'Hello'}]

class AsyncTestPassModel(unittest.IsolatedAsyncioTestCase):

    async def test_response(self):
        client = Client(provider=AsyncGeneratorProviderMock)
        response = await client.chat.completions.async_create(DEFAULT_MESSAGES, "")
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("Mock", response.choices[0].message.content)

    async def test_pass_model(self):
        client = Client(provider=ModelProviderMock)
        response = await client.chat.completions.async_create(DEFAULT_MESSAGES, "Hello")
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("Hello", response.choices[0].message.content)

    async def test_max_tokens(self):
        client = Client(provider=YieldProviderMock)
        messages = [{'role': 'user', 'content': chunk} for chunk in ["How ", "are ", "you", "?"]]
        response = await client.chat.completions.async_create(messages, "Hello", max_tokens=1)
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("How ", response.choices[0].message.content)
        response = await client.chat.completions.async_create(messages, "Hello", max_tokens=2)
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("How are ", response.choices[0].message.content)

    async def test_max_stream(self):
        client = Client(provider=YieldProviderMock)
        messages = [{'role': 'user', 'content': chunk} for chunk in ["How ", "are ", "you", "?"]]
        response = await client.chat.completions.async_create(messages, "Hello", stream=True)
        async for chunk in response:
            self.assertIsInstance(chunk, ChatCompletionChunk)
            if chunk.choices[0].delta.content is not None:
                self.assertIsInstance(chunk.choices[0].delta.content, str)
        messages = [{'role': 'user', 'content': chunk} for chunk in ["You ", "You ", "Other", "?"]]
        response = await client.chat.completions.async_create(messages, "Hello", stream=True, max_tokens=2)
        response_list = []
        async for chunk in response:
            response_list.append(chunk)
        self.assertEqual(len(response_list), 3)
        for chunk in response_list:
            if chunk.choices[0].delta.content is not None:
                self.assertEqual(chunk.choices[0].delta.content, "You ")

    async def test_stop(self):
        client = Client(provider=YieldProviderMock)
        messages = [{'role': 'user', 'content': chunk} for chunk in ["How ", "are ", "you", "?"]]
        response = await client.chat.completions.async_create(messages, "Hello", stop=["and"])
        self.assertIsInstance(response, ChatCompletion)
        self.assertEqual("How are you?", response.choices[0].message.content)

if __name__ == '__main__':
    unittest.main()
