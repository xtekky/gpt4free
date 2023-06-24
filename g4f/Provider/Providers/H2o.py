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

models = {
    'falcon-7b': 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3',
    'falcon-40b': 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v1',
    'llama-13b': 'h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-13b'
}

def _create_completion(model: str, messages: list, stream: bool, **kwargs):
    conversation = 'instruction: this is a conversation beween, a user and an AI assistant, respond to the latest message, referring to the conversation if needed\n'
    for message in messages:
        conversation += '%s: %s\n' % (message['role'], message['content'])
    conversation += 'assistant:'
    
    client = Session()
    client.headers = {
        'authority': 'gpt-gm.h2o.ai',
        'origin': 'https://gpt-gm.h2o.ai',
        'referer': 'https://gpt-gm.h2o.ai/',
        'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    }

    client.get('https://gpt-gm.h2o.ai/')
    response = client.post('https://gpt-gm.h2o.ai/settings', data={
        'ethicsModalAccepted': 'true',
        'shareConversationsWithModelAuthors': 'true',
        'ethicsModalAcceptedAt': '',
        'activeModel': 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v1',
        'searchEnabled': 'true',
    })

    headers = {
        'authority': 'gpt-gm.h2o.ai',
        'accept': '*/*',
        'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
        'origin': 'https://gpt-gm.h2o.ai',
        'referer': 'https://gpt-gm.h2o.ai/',
        'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    }

    json_data = {
        'model': models[model]
    }

    response = client.post('https://gpt-gm.h2o.ai/conversation',
                            headers=headers, json=json_data)
    conversationId = response.json()['conversationId']


    completion = client.post(f'https://gpt-gm.h2o.ai/conversation/{conversationId}', stream=True, json = {
        'inputs': conversation,
        'parameters': {
            'temperature': kwargs.get('temperature', 0.4),
            'truncate': kwargs.get('truncate', 2048),
            'max_new_tokens': kwargs.get('max_new_tokens', 1024),
            'do_sample': kwargs.get('do_sample', True),
            'repetition_penalty': kwargs.get('repetition_penalty', 1.2),
            'return_full_text': kwargs.get('return_full_text', False)
        },
        'stream': True,
        'options': {
            'id': kwargs.get('id', str(uuid4())),
            'response_id': kwargs.get('response_id', str(uuid4())),
            'is_retry': False,
            'use_cache': False,
            'web_search_id': ''
        }
    })

    for line in completion.iter_lines():
        if b'data' in line:
            line = loads(line.decode('utf-8').replace('data:', ''))
            token = line['token']['text']
            
            if token == '<|endoftext|>':
                break
            else:
                yield (token)
            
params = f'g4f.Providers.{os.path.basename(__file__)[:-3]} supports: ' + \
    '(%s)' % ', '.join([f"{name}: {get_type_hints(_create_completion)[name].__name__}" for name in _create_completion.__code__.co_varnames[:_create_completion.__code__.co_argcount]])