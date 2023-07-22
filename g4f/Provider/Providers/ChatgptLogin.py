import os
from ...typing import sha256, Dict, get_type_hints
import requests
import re
import base64

url = 'https://chatgptlogin.ac'
model = ['gpt-3.5-turbo']
supports_stream = False
needs_auth = False
working = False

def _create_completion(model: str, messages: list, stream: bool, **kwargs):
    def get_nonce():
        res = requests.get('https://chatgptlogin.ac/use-chatgpt-free/', headers={
            "Referer": "https://chatgptlogin.ac/use-chatgpt-free/",
            "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        })

        src = re.search(r'class="mwai-chat mwai-chatgpt">.*<span>Send</span></button></div></div></div> <script defer src="(.*?)">', res.text).group(1)
        decoded_string = base64.b64decode(src.split(",")[-1]).decode('utf-8')
        return re.search(r"let restNonce = '(.*?)';", decoded_string).group(1)
    
    def transform(messages: list) -> list:
        def html_encode(string: str) -> str:
            table = {
                '"': '&quot;',
                "'": '&#39;',
                '&': '&amp;',
                '>': '&gt;',
                '<': '&lt;',
                '\n': '<br>',
                '\t': '&nbsp;&nbsp;&nbsp;&nbsp;',
                ' ': '&nbsp;'
            }
            
            for key in table:
                string = string.replace(key, table[key])
                
            return string
        
        return [{
            'id': os.urandom(6).hex(),
            'role': message['role'],
            'content': message['content'],
            'who': 'AI: ' if message['role'] == 'assistant' else 'User: ',
            'html': html_encode(message['content'])} for message in messages]
    
    headers = {
        'authority': 'chatgptlogin.ac',
        'accept': '*/*',
        'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
        'content-type': 'application/json',
        'origin': 'https://chatgptlogin.ac',
        'referer': 'https://chatgptlogin.ac/use-chatgpt-free/',
        'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'x-wp-nonce': get_nonce()
    }
    
    conversation = transform(messages)

    json_data = {
        'env': 'chatbot',
        'session': 'N/A',
        'prompt': 'Converse as if you were an AI assistant. Be friendly, creative.',
        'context': 'Converse as if you were an AI assistant. Be friendly, creative.',
        'messages': conversation,
        'newMessage': messages[-1]['content'],
        'userName': '<div class="mwai-name-text">User:</div>',
        'aiName': '<div class="mwai-name-text">AI:</div>',
        'model': 'gpt-3.5-turbo',
        'temperature': kwargs.get('temperature', 0.8),
        'maxTokens': 1024,
        'maxResults': 1,
        'apiKey': '',
        'service': 'openai',
        'embeddingsIndex': '',
        'stop': '',
        'clientId': os.urandom(6).hex()
    }

    response = requests.post('https://chatgptlogin.ac/wp-json/ai-chatbot/v1/chat', 
                             headers=headers, json=json_data)
    
    return response.json()['reply']


params = f'g4f.Providers.{os.path.basename(__file__)[:-3]} supports: ' + \
    '(%s)' % ', '.join(
        [f"{name}: {get_type_hints(_create_completion)[name].__name__}" for name in _create_completion.__code__.co_varnames[:_create_completion.__code__.co_argcount]])
