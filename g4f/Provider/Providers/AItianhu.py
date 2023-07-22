import os, requests
from ...typing import sha256, Dict, get_type_hints
import json

url = "https://www.aitianhu.com/api/chat-process"
model = ['gpt-3.5-turbo']
supports_stream = False
needs_auth = False
working = True


def _create_completion(model: str, messages: list, stream: bool, **kwargs):
    base = ''
    for message in messages:
        base += '%s: %s\n' % (message['role'], message['content'])
    base += 'assistant:'

    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    data = {
        "prompt": base,
        "options": {},
        "systemMessage": "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown.",
        "temperature": kwargs.get("temperature", 0.8),
        "top_p": kwargs.get("top_p", 1)
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        lines = response.text.strip().split('\n')
        res = json.loads(lines[-1])
        yield res['text']
    else:
        print(f"Error Occurred::{response.status_code}")
        return None

params = f'g4f.Providers.{os.path.basename(__file__)[:-3]} supports: ' + \
    '(%s)' % ', '.join([f"{name}: {get_type_hints(_create_completion)[name].__name__}" for name in _create_completion.__code__.co_varnames[:_create_completion.__code__.co_argcount]])