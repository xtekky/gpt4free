from __future__ import annotations

from ..typing import Messages, CreateResult
from ..providers.base_provider import AbstractProvider, ProviderModelMixin

import time, uuid, random, json
from requests import Session

from .openai.new import (
    get_config,
    get_answer_token,
    process_turnstile,
    get_requirements_token
)

def format_conversation(messages: list):
    conversation = []
    
    for message in messages:
        conversation.append({
            'id': str(uuid.uuid4()),
            'author': {
                'role': message['role'],
            },
            'content': {
                'content_type': 'text',
                'parts': [
                    message['content'],
                ],
            },
            'metadata': {
                'serialization_metadata': {
                    'custom_symbol_offsets': [],
                },
            },
            'create_time': round(time.time(), 3),
        })
    
    return conversation

def init_session(user_agent):
    session = Session()

    cookies = {
        '_dd_s': '',
    }

    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.8',
        'cache-control': 'no-cache',
        'pragma': 'no-cache',
        'priority': 'u=0, i',
        'sec-ch-ua': '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
        'sec-ch-ua-arch': '"arm"',
        'sec-ch-ua-bitness': '"64"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-model': '""',
        'sec-ch-ua-platform': '"macOS"',
        'sec-ch-ua-platform-version': '"14.4.0"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': user_agent,
    }

    session.get('https://chatgpt.com/', cookies=cookies, headers=headers)
    
    return session

class ChatGpt(AbstractProvider, ProviderModelMixin):
    label = "ChatGpt"
    working = True
    supports_message_history = True
    supports_system_message = True
    supports_stream = True
    models = [
        'gpt-4o',
        'gpt-4o-mini',
        'gpt-4',
        'gpt-4-turbo',
        'chatgpt-4o-latest',
    ]

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        **kwargs
    ) -> CreateResult:
        
        if model in [
            'gpt-4o',
            'gpt-4o-mini',
            'gpt-4',
            'gpt-4-turbo',
            'chatgpt-4o-latest'
        ]:
            model = 'auto'
        
        elif model in [
            'gpt-3.5-turbo'
        ]:
            model = 'text-davinci-002-render-sha'
            
        else:
            raise ValueError(f"Invalid model: {model}")
        
        user_agent       = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'
        session: Session = init_session(user_agent)
        
        config   = get_config(user_agent)
        pow_req  = get_requirements_token(config)
        headers  = { 
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.8',
            'content-type': 'application/json',
            'oai-device-id': f'{uuid.uuid4()}',
            'oai-language': 'en-US',
            'origin': 'https://chatgpt.com',
            'priority': 'u=1, i',
            'referer': 'https://chatgpt.com/',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'sec-gpc': '1',
            'user-agent': f'{user_agent}'
        }
        
        response = session.post('https://chatgpt.com/backend-anon/sentinel/chat-requirements',
            headers=headers, json={'p': pow_req})

        if response.status_code != 200:
            print(f"Request failed with status: {response.status_code}")
            print(f"Response content: {response.content}")
            return

        response_data = response.json()
        if "detail" in response_data and "Unusual activity" in response_data["detail"]:
            print(f"Blocked due to unusual activity: {response_data['detail']}")
            return
        
        turnstile          = response_data.get('turnstile', {})
        turnstile_required = turnstile.get('required')
        pow_conf           = response_data.get('proofofwork', {})

        if turnstile_required:
            turnstile_dx    = turnstile.get('dx')
            turnstile_token = process_turnstile(turnstile_dx, pow_req)
        
        headers = headers | {
            'openai-sentinel-turnstile-token'        : turnstile_token,
            'openai-sentinel-chat-requirements-token': response_data.get('token'),
            'openai-sentinel-proof-token'            : get_answer_token(
                pow_conf.get('seed'), pow_conf.get('difficulty'), config
            )
        }
        
        json_data = {
            'action': 'next',
            'messages': format_conversation(messages),
            'parent_message_id': str(uuid.uuid4()),
            'model': 'auto',
            'timezone_offset_min': -120,
            'suggestions': [
                'Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.',
                'Could you help me plan a relaxing day that focuses on activities for rejuvenation? To start, can you ask me what my favorite forms of relaxation are?',
                'I have a photoshoot tomorrow. Can you recommend me some colors and outfit options that will look good on camera?',
                'Make up a 5-sentence story about "Sharky", a tooth-brushing shark superhero. Make each sentence a bullet point.',
            ],
            'history_and_training_disabled': False,
            'conversation_mode': {
                'kind': 'primary_assistant',
            },
            'force_paragen': False,
            'force_paragen_model_slug': '',
            'force_nulligen': False,
            'force_rate_limit': False,
            'reset_rate_limits': False,
            'websocket_request_id': str(uuid.uuid4()),
            'system_hints': [],
            'force_use_sse': True,
            'conversation_origin': None,
            'client_contextual_info': {
                'is_dark_mode': True,
                'time_since_loaded': random.randint(22,33),
                'page_height': random.randint(600, 900),
                'page_width': random.randint(500, 800),
                'pixel_ratio': 2,
                'screen_height': random.randint(800, 1200),
                'screen_width': random.randint(1200, 2000),
            },
        }

        time.sleep(2)
        
        response = session.post('https://chatgpt.com/backend-anon/conversation',
            headers=headers, json=json_data, stream=True)

        replace = ''
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode()
                print(f"Received line: {decoded_line}")
                if decoded_line.startswith('data:'):
                    json_string = decoded_line[6:]
                    if json_string.strip():
                        try:
                            data = json.loads(json_string)
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}, content: {json_string}")
                            continue
                        
                        if data.get('message').get('author').get('role') == 'assistant':
                            tokens = (data.get('message').get('content').get('parts')[0])
                            
                            yield tokens.replace(replace, '')
                            
                            replace = tokens
