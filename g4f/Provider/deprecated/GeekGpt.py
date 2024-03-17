from __future__ import annotations
import requests, json

from ..base_provider import AbstractProvider
from ...typing       import CreateResult, Messages
from json           import dumps


class GeekGpt(AbstractProvider):
    url = 'https://chat.geekgpt.org'
    working = False
    supports_message_history = True
    supports_stream = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        **kwargs
    ) -> CreateResult:
        if not model:
            model = "gpt-3.5-turbo"
        json_data = {
            'messages': messages,
            'model': model,
            'temperature': kwargs.get('temperature', 0.9),
            'presence_penalty': kwargs.get('presence_penalty', 0),
            'top_p': kwargs.get('top_p', 1),
            'frequency_penalty': kwargs.get('frequency_penalty', 0),
            'stream': True
        }

        data = dumps(json_data, separators=(',', ':'))

        headers = {
            'authority': 'ai.fakeopen.com',
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'authorization': 'Bearer pk-this-is-a-real-free-pool-token-for-everyone',
            'content-type': 'application/json',
            'origin': 'https://chat.geekgpt.org',
            'referer': 'https://chat.geekgpt.org/',
            'sec-ch-ua': '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'cross-site',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        }

        response = requests.post("https://ai.fakeopen.com/v1/chat/completions", 
                                 headers=headers, data=data, stream=True)
        response.raise_for_status()

        for chunk in response.iter_lines():
            if b'content' in chunk:
                json_data = chunk.decode().replace("data: ", "")

                if json_data == "[DONE]":
                    break
                
                try:
                    content = json.loads(json_data)["choices"][0]["delta"].get("content")
                except Exception as e:
                    raise RuntimeError(f'error | {e} :', json_data)
                
                if content:
                    yield content