import requests
import os
import json
from ...typing import sha256, Dict, get_type_hints
url = 'http://supertest.lockchat.app'
model = ['gpt-4', 'gpt-3.5-turbo']
supports_stream = True
needs_auth = False

def _create_completion(model: str, messages: list, stream: bool, temperature: float = 0.7, **kwargs):

    payload = {
        "temperature": 0.7,
        "messages": messages,
        "model": model,
        "stream": True,
    }
    headers = {
        "user-agent": "ChatX/39 CFNetwork/1408.0.4 Darwin/22.5.0",
    }
    response = requests.post("http://supertest.lockchat.app/v1/chat/completions", 
                            json=payload, headers=headers, stream=True)
    for token in response.iter_lines():
        if b'The model: `gpt-4` does not exist' in token:
            print('error, retrying...')
            _create_completion(model=model, messages=messages, stream=stream, temperature=temperature, **kwargs)
        if b"content" in token:
            token = json.loads(token.decode('utf-8').split('data: ')[1])['choices'][0]['delta'].get('content')
            if token: yield (token)
            
params = f'g4f.Providers.{os.path.basename(__file__)[:-3]} supports: ' + \
    '(%s)' % ', '.join([f"{name}: {get_type_hints(_create_completion)[name].__name__}" for name in _create_completion.__code__.co_varnames[:_create_completion.__code__.co_argcount]])