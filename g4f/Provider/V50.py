import os, uuid, requests
from ..typing import get_type_hints
url = 'https://p5.v50.ltd'
model = ['gpt-3.5-turbo','gpt-3.5-turbo-16k']
supports_stream = False
needs_auth = False
working = True

def _create_completion(model: str, messages: list, stream: bool, temperature: float = 0.7, **kwargs):

    conversation = ''
    for message in messages:
        conversation += '%s: %s\n' % (message['role'], message['content'])
    
    conversation += 'assistant: '
    payload = {
        "prompt": conversation,
        "options": {},
        "systemMessage": ".",
        "temperature": temperature,
        "top_p": 1,
        "model": model,
        "user": str(uuid.uuid4())
    }
    headers = {
        'authority': 'p5.v50.ltd',
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
        'content-type': 'application/json',
        'origin': 'https://p5.v50.ltd',
        'referer': 'https://p5.v50.ltd/',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    response = requests.post("https://p5.v50.ltd/api/chat-process", 
                            json=payload, headers=headers, proxies=kwargs['proxy'] if 'proxy' in kwargs else {})
    yield response.text
            
params = f'g4f.Providers.{os.path.basename(__file__)[:-3]} supports: ' + \
    '(%s)' % ', '.join([f"{name}: {get_type_hints(_create_completion)[name].__name__}" for name in _create_completion.__code__.co_varnames[:_create_completion.__code__.co_argcount]])