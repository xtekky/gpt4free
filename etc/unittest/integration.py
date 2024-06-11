import unittest
import json

try:
    import nest_asyncio
    has_nest_asyncio = True
except ImportError:
    has_nest_asyncio = False 

from g4f.client import Client, ChatCompletion
from g4f.Provider import Bing, OpenaiChat

DEFAULT_MESSAGES = [{"role": "system", "content": 'Response in json, Example: {"success": false}'},
                    {"role": "user", "content": "Say success true in json"}]

class TestProviderIntegration(unittest.TestCase):
    def setUp(self):
        if not has_nest_asyncio:
            self.skipTest("nest_asyncio is not installed")

    def test_bing(self):
        self.skipTest("Not working")
        client = Client(provider=Bing)
        response = client.chat.completions.create(DEFAULT_MESSAGES, "", response_format={"type": "json_object"})
        self.assertIsInstance(response, ChatCompletion)
        self.assertIn("success", json.loads(response.choices[0].message.content))

    def test_openai(self):
        self.skipTest("not working in this network")
        client = Client(provider=OpenaiChat)
        response = client.chat.completions.create(DEFAULT_MESSAGES, "", response_format={"type": "json_object"})
        self.assertIsInstance(response, ChatCompletion)
        self.assertIn("success", json.loads(response.choices[0].message.content))

if __name__ == '__main__':
    unittest.main()