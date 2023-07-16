import os, requests
from ...typing import sha256, Dict, get_type_hints
import json

url = "https://chat.acytoo.com/api/completions"
model = ['gpt-3.5-turbo']
supports_stream = False
needs_auth = False
working = False

def _create_completion(model: str, messages: list, stream: bool, **kwargs):
    base = ''
    for message in messages:
        base += '%s: %s\n' % (message['role'], message['content'])
    base += 'assistant:'

    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    data = {
        "key": "",
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": base,
                "createdAt": 1688518523500
            }
        ],
        "temperature": 1,
        "password": ""
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        yield response.text
    else:
        print(f"Error Occurred::{response.status_code}")
        return None

params = f'g4f.Providers.{os.path.basename(__file__)[:-3]} supports: ' + \
    '(%s)' % ', '.join([f"{name}: {get_type_hints(_create_completion)[name].__name__}" for name in _create_completion.__code__.co_varnames[:_create_completion.__code__.co_argcount]])