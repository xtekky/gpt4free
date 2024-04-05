import unittest
import json

from g4f.client import Client, ChatCompletion
from g4f.Provider import Bing, OpenaiChat

DEFAULT_MESSAGES = [{"role": "system", "content": 'Response in json, Example: {"success: True"}'},
                    {"role": "user", "content": "Say success true in json"}]

class TestProviderIntegration(unittest.TestCase):

    def test_bing(self):
        client = Client(provider=Bing)
        response = client.chat.completions.create(DEFAULT_MESSAGES, "", response_format={"type": "json_object"})
        self.assertIsInstance(response, ChatCompletion)
        self.assertIn("success", json.loads(response.choices[0].message.content))

    def test_openai(self):
        client = Client(provider=OpenaiChat)
        response = client.chat.completions.create(DEFAULT_MESSAGES, "", response_format={"type": "json_object"})
        self.assertIsInstance(response, ChatCompletion)
        self.assertIn("success", json.loads(response.choices[0].message.content))

if __name__ == '__main__':
    unittest.main()