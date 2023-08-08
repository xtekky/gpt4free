import os, requests
from ...typing import sha256, Dict, get_type_hints
import json

url = "https://free.easychat.work"
model = ['gpt-3.5-turbo']
supports_stream = True
needs_auth = False
working = True


def _create_completion(model: str, messages: list, stream: bool, **kwargs):
    req = requests.Session()
    
    proxy = {
        "http": "http://159.89.138.130:80"
    }
    
    headers = {
        'authority': 'beta.easychat.work',
        'accept': 'text/event-stream',
        'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3,fa=0.2',
        'content-type': 'application/json',
        'origin': 'https://beta.easychat.work',
        'referer': 'https://beta.easychat.work/',
        'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest',
    }

    json_data = {
        'messages': messages,
        'stream': True,
        'model': "gpt-3.5-turbo",
        'temperature': kwargs.get('temperature', 0.5),
        'presence_penalty': kwargs.get('presence_penalty', 0),
        'frequency_penalty': kwargs.get('frequency_penalty', 0),
        'top_p': kwargs.get('top_p', 1),
    }

    # init cookies from server
    req.get("https://site.easygpt.work/", proxies=proxy)

    response = req.post('https://beta.easychat.work/api/openai/v1/chat/completions',
        headers=headers, json=json_data, proxies=proxy)
    
    for chunk in response.iter_lines():
        if b'content' in chunk:
            data = json.loads(chunk.decode().split('data: ')[1])
            yield (data['choices'][0]['delta']['content'])

params = f'g4f.Providers.{os.path.basename(__file__)[:-3]} supports: ' + \
    '(%s)' % ', '.join([f"{name}: {get_type_hints(_create_completion)[name].__name__}" for name in _create_completion.__code__.co_varnames[:_create_completion.__code__.co_argcount]])
