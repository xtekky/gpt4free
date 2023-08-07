import os
import requests
from ...typing import sha256, Dict, get_type_hints

url = 'https://opchatgpts.net'
model = ['gpt-3.5-turbo']
supports_stream = False
needs_auth = False
working = True

def _create_completion(model: str, messages: list, stream: bool = False, temperature: float = 0.8, max_tokens: int = 1024, system_prompt: str = "Converse as if you were an AI assistant. Be friendly, creative.", **kwargs):

    data = {
        'env': 'chatbot',
        'session': 'N/A',
        'prompt': "\n",
        'context': system_prompt,
        'messages': messages,
        'newMessage': messages[::-1][0]["content"],
        'userName': '<div class="mwai-name-text">User:</div>',
        'aiName': '<div class="mwai-name-text">AI:</div>',
        'model': 'gpt-3.5-turbo',
        'temperature': temperature,
        'maxTokens': max_tokens,
        'maxResults': 1,
        'apiKey': '',
        'service': 'openai',
        'embeddingsIndex': '',
        'stop': ''
    }

    response = requests.post('https://opchatgpts.net/wp-json/ai-chatbot/v1/chat', json=data).json()

    if response["success"]:

        return response["reply"] # `yield (response["reply"])` doesn't work

    raise Exception("Request failed: " + response)

params = f'g4f.Providers.{os.path.basename(__file__)[:-3]} supports: ' + \
    '(%s)' % ', '.join(
        [f"{name}: {get_type_hints(_create_completion)[name].__name__}" for name in _create_completion.__code__.co_varnames[:_create_completion.__code__.co_argcount]])
