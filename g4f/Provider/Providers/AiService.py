import os,sys
import requests
from ...typing import get_type_hints

url = "https://aiservice.vercel.app/api/chat/answer"
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
        "accept": "*/*",
        "content-type": "text/plain;charset=UTF-8",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "Referer": "https://aiservice.vercel.app/chat",
    }
    data = {
        "input": base
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        _json = response.json()
        yield _json['data']
    else:
        print(f"Error Occurred::{response.status_code}")
        return None
    


params = f'g4f.Providers.{os.path.basename(__file__)[:-3]} supports: ' + \
    '(%s)' % ', '.join(
        [f"{name}: {get_type_hints(_create_completion)[name].__name__}" for name in _create_completion.__code__.co_varnames[:_create_completion.__code__.co_argcount]])