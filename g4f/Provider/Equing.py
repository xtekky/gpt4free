import requests, json

from ..typing import Any, CreateResult
from .base_provider import BaseProvider


class Equing(BaseProvider):
    url: str = 'https://next.eqing.tech/'
    working = True
    needs_auth = False
    supports_stream = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = False

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs: Any) -> CreateResult:

        headers = {
            'authority': 'next.eqing.tech',
            'accept': 'text/event-stream',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'cache-control': 'no-cache',
            'content-type': 'application/json',
            'origin': 'https://next.eqing.tech',
            'plugins': '0',
            'pragma': 'no-cache',
            'referer': 'https://next.eqing.tech/',
            'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'usesearch': 'false',
            'x-requested-with': 'XMLHttpRequest',
        }
        json_data = {
            'messages': messages,
            'stream': True,
            'model': model.split(":",1)[-1],
            'temperature': kwargs.get('temperature', 0.5),
            'presence_penalty': kwargs.get('presence_penalty', 0),
            'frequency_penalty': kwargs.get('frequency_penalty', 0),
            'top_p': kwargs.get('top_p', 1),
        }

        response = requests.post('https://next.eqing.tech/api/openai/v1/chat/completions', headers=headers, json=json_data, stream=True)
        response.raise_for_status()
        return _process_response(response)


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
    
def _process_response(response):
    buffer = ""
    for chunk in response.iter_content(chunk_size=None):
        decoded_chunk = chunk.decode('utf-8')
        buffer += decoded_chunk
        while "\n\n" in buffer:
            event_data, buffer = buffer.split("\n\n", 1)
            event_data = event_data.strip()
            if event_data.startswith("data:"):
                event_data = event_data[5:].strip()
                try:
                    data = json.loads(event_data)
                    choice = data.get('choices', [{}])[0]
                    finish_reason = choice.get('finish_reason')
                    content = choice.get('delta', {}).get('content')    
                    if finish_reason != 'stop' and content:
                        yield content
                    elif finish_reason == 'stop':
                        return
                except json.JSONDecodeError:
                    pass
