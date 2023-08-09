import json
import os

import requests
from g4f.typing import get_type_hints

url = "https://backend.raycast.com/api/v1/ai/chat_completions"
model = ['gpt-3.5-turbo', 'gpt-4']
supports_stream = True
needs_auth = True
working = True


def _create_completion(model: str, messages: list, stream: bool, **kwargs):
    auth = kwargs.get('auth')
    headers = {
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
        'Authorization': f'Bearer {auth}',
        'Content-Type': 'application/json',
        'User-Agent': 'Raycast/0 CFNetwork/1410.0.3 Darwin/22.6.0',
    }
    parsed_messages = []
    for message in messages:
        parsed_messages.append({
            'author': message['role'],
            'content': {'text': message['content']}
        })
    data = {
        "debug": False,
        "locale": "en-CN",
        "messages": parsed_messages,
        "model": model,
        "provider": "openai",
        "source": "ai_chat",
        "system_instruction": "markdown",
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, json=data, stream=True)
    for token in response.iter_lines():
        if b'data: ' not in token:
            continue
        completion_chunk = json.loads(token.decode().replace('data: ', ''))
        token = completion_chunk['text']
        if token != None:
            yield token


params = f'g4f.Providers.{os.path.basename(__file__)[:-3]} supports: ' + \
         '(%s)' % ', '.join([f"{name}: {get_type_hints(_create_completion)[name].__name__}" for name in _create_completion.__code__.co_varnames[:_create_completion.__code__.co_argcount]])
