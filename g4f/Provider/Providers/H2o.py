from requests import Session
from uuid import uuid4
from json import loads
import os
import json
import requests
from ...typing import sha256, Dict, get_type_hints

url = 'https://gpt-gm.h2o.ai'
model = ['falcon-40b', 'falcon-7b', 'llama-13b']
supports_stream = True
needs_auth = False
working = True

models = {
    'falcon-7b': 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3',
    'falcon-40b': 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v1',
    'llama-13b': 'h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-13b'
}

def _create_completion(model: str, messages: list, stream: bool, **kwargs):
    
    conversation = ''
    for message in messages:
        conversation += '%s: %s\n' % (message['role'], message['content'])
    
    conversation += 'assistant: '
    session = requests.Session()

    response = session.get("https://gpt-gm.h2o.ai/")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3",
        "Content-Type": "application/x-www-form-urlencoded",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Referer": "https://gpt-gm.h2o.ai/r/jGfKSwU"
    }
    data = {
        "ethicsModalAccepted": "true",
        "shareConversationsWithModelAuthors": "true",
        "ethicsModalAcceptedAt": "",
        "activeModel": "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v1",
        "searchEnabled": "true"
    }
    response = session.post("https://gpt-gm.h2o.ai/settings", headers=headers, data=data)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Accept": "*/*",
        "Accept-Language": "ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3",
        "Content-Type": "application/json",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Referer": "https://gpt-gm.h2o.ai/"
    }
    data = {
        "model": models[model]
    }
    
    conversation_id = session.post("https://gpt-gm.h2o.ai/conversation", headers=headers, json=data)
    data = {
        "inputs": conversation,
        "parameters": {
            "temperature": kwargs.get('temperature', 0.4),
            "truncate": kwargs.get('truncate', 2048),
            "max_new_tokens": kwargs.get('max_new_tokens', 1024),
            "do_sample": kwargs.get('do_sample', True),
            "repetition_penalty": kwargs.get('repetition_penalty', 1.2),
            "return_full_text": kwargs.get('return_full_text', False)
        },
        "stream": True,
        "options": {
            "id": kwargs.get('id', str(uuid4())),
            "response_id": kwargs.get('response_id', str(uuid4())),
            "is_retry": False,
            "use_cache": False,
            "web_search_id": ""
        }
    }
    
    response = session.post(f"https://gpt-gm.h2o.ai/conversation/{conversation_id.json()['conversationId']}", headers=headers, json=data)
    generated_text = response.text.replace("\n", "").split("data:")
    generated_text = json.loads(generated_text[-1])
    
    return generated_text["generated_text"]

params = f'g4f.Providers.{os.path.basename(__file__)[:-3]} supports: ' + \
    '(%s)' % ', '.join([f"{name}: {get_type_hints(_create_completion)[name].__name__}" for name in _create_completion.__code__.co_varnames[:_create_completion.__code__.co_argcount]])