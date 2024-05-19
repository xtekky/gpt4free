from __future__ import annotations

import json, random, requests, threading
from aiohttp import ClientSession

from ..typing import CreateResult, Messages
from .base_provider import AbstractProvider
from .helper import format_prompt

class Cohere(AbstractProvider):
    url                   = "https://cohereforai-c4ai-command-r-plus.hf.space"
    working               = False
    supports_gpt_35_turbo = False
    supports_gpt_4        = False
    supports_stream       = True
    
    @staticmethod
    def create_completion(
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        max_retries: int = 6,
        **kwargs
    ) -> CreateResult:
        
        prompt = format_prompt(messages)
        
        headers = {
            'accept': 'text/event-stream',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'cache-control': 'no-cache',
            'pragma': 'no-cache',
            'referer': 'https://cohereforai-c4ai-command-r-plus.hf.space/?__theme=light',
            'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        }
        
        session_hash = ''.join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=11))

        params = {
            'fn_index': '1',
            'session_hash': session_hash,
        }

        response = requests.get(
            'https://cohereforai-c4ai-command-r-plus.hf.space/queue/join',
            params=params,
            headers=headers,
            stream=True
        )
        
        completion = ''

        for line in response.iter_lines():
            if line:
                json_data = json.loads(line[6:])
                
                if b"send_data" in (line):
                    event_id = json_data["event_id"]
                    
                    threading.Thread(target=send_data, args=[session_hash, event_id, prompt]).start()
                
                if b"process_generating" in line or b"process_completed" in line:
                    token = (json_data['output']['data'][0][0][1])
                    
                    yield (token.replace(completion, ""))
                    completion = token

def send_data(session_hash, event_id, prompt):
    headers = {
        'accept': '*/*',
        'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
        'cache-control': 'no-cache',
        'content-type': 'application/json',
        'origin': 'https://cohereforai-c4ai-command-r-plus.hf.space',
        'pragma': 'no-cache',
        'referer': 'https://cohereforai-c4ai-command-r-plus.hf.space/?__theme=light',
        'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    }

    json_data = {
        'data': [
            prompt,
            '',
            [],
        ],
        'event_data': None,
        'fn_index': 1,
        'session_hash': session_hash,
        'event_id': event_id
    }
    
    requests.post('https://cohereforai-c4ai-command-r-plus.hf.space/queue/data',
                    json = json_data, headers=headers)