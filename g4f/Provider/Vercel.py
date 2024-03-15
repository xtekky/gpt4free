from __future__ import annotations

import json, base64, requests, random, os

try:
    import execjs
    has_requirements = True
except ImportError:
    has_requirements = False

from ..typing       import Messages, CreateResult
from .base_provider import AbstractProvider
from ..requests     import raise_for_status
from ..errors       import MissingRequirementsError, RateLimitError, ResponseStatusError

class Vercel(AbstractProvider):
    url = 'https://chat.vercel.ai'
    working = True
    supports_message_history = True
    supports_system_message  = True
    supports_gpt_35_turbo    = True
    supports_stream          = True
    
    @staticmethod
    def create_completion(
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        max_retries: int = 6,
        **kwargs
    ) -> CreateResult:
        if not has_requirements:
            raise MissingRequirementsError('Install "PyExecJS" package')
        
        headers = {
            'authority': 'chat.vercel.ai',
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'cache-control': 'no-cache',
            'content-type': 'application/json',
            'custom-encoding': get_anti_bot_token(),
            'origin': 'https://chat.vercel.ai',
            'pragma': 'no-cache',
            'referer': 'https://chat.vercel.ai/',
            'sec-ch-ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        }

        json_data = {
            'messages': messages,
            'id'      : f'{os.urandom(3).hex()}a',
        }
        response = None
        for _ in range(max_retries):
            response = requests.post('https://chat.vercel.ai/api/chat', 
                                    headers=headers, json=json_data, stream=True, proxies={"https": proxy})
            if not response.ok:
                continue
            for token in response.iter_content(chunk_size=None):
                try:
                    yield token.decode()
                except UnicodeDecodeError:
                    pass
            break
        raise_for_status(response)

def get_anti_bot_token() -> str:
    headers = {
        'authority': 'sdk.vercel.ai',
        'accept': '*/*',
        'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
        'cache-control': 'no-cache',
        'pragma': 'no-cache',
        'referer': 'https://sdk.vercel.ai/',
        'sec-ch-ua': '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': f'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.{random.randint(99, 999)}.{random.randint(99, 999)} Safari/537.36',
    }

    response = requests.get('https://chat.vercel.ai/openai.jpeg', 
                            headers=headers).text

    raw_data = json.loads(base64.b64decode(response, 
                                    validate=True))

    js_script = '''const globalThis={marker:"mark"};String.prototype.fontcolor=function(){return `<font>${this}</font>`};
        return (%s)(%s)''' % (raw_data['c'], raw_data['a'])

    sec_list = [execjs.compile(js_script).call('')[0], [], "sentinel"]

    raw_token = json.dumps({'r': sec_list, 't': raw_data['t']}, 
                        separators = (",", ":"))

    return base64.b64encode(raw_token.encode('utf-8')).decode()