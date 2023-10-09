from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod

import requests

from ...typing import Any, CreateResult
from ..base_provider import BaseProvider


class FastGpt(BaseProvider):
    url: str                = 'https://chat9.fastgpt.me/'
    working                 = False
    needs_auth              = False
    supports_stream         = True
    supports_gpt_35_turbo   = True
    supports_gpt_4          = False

    @staticmethod
    @abstractmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:

        headers = {
            'authority'         : 'chat9.fastgpt.me',
            'accept'            : 'text/event-stream',
            'accept-language'   : 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'cache-control'     : 'no-cache',
            'content-type'      : 'application/json',
            'origin'            : 'https://chat9.fastgpt.me',
            'plugins'           : '0',
            'pragma'            : 'no-cache',
            'referer'           : 'https://chat9.fastgpt.me/',
            'sec-ch-ua'         : '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
            'sec-ch-ua-mobile'  : '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest'    : 'empty',
            'sec-fetch-mode'    : 'cors',
            'sec-fetch-site'    : 'same-origin',
            'user-agent'        : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'usesearch'         : 'false',
            'x-requested-with'  : 'XMLHttpRequest',
        }

        json_data = {
            'messages'          : messages,
            'stream'            : stream,
            'model'             : model,
            'temperature'       : kwargs.get('temperature', 0.5),
            'presence_penalty'  : kwargs.get('presence_penalty', 0),
            'frequency_penalty' : kwargs.get('frequency_penalty', 0),
            'top_p'             : kwargs.get('top_p', 1),
        }
        
        subdomain = random.choice([
            'jdaen979ew',
            'chat9'
        ])

        response = requests.post(f'https://{subdomain}.fastgpt.me/api/openai/v1/chat/completions',
                                 headers=headers, json=json_data, stream=stream)
        
        for line in response.iter_lines():
            if line:
                try:
                    if b'content' in line:
                            line_json = json.loads(line.decode('utf-8').split('data: ')[1])
                            token = line_json['choices'][0]['delta'].get('content')
                            if token:
                                yield token
                except:
                    continue

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"