from __future__ import annotations

import json, requests, re

from curl_cffi      import requests as cf_reqs
from ..typing       import CreateResult, Messages
from .base_provider import ProviderModelMixin, AbstractProvider
from .helper        import format_prompt

class HuggingChat(AbstractProvider, ProviderModelMixin):
    url             = "https://huggingface.co/chat"
    working         = True
    supports_stream = True
    default_model   = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    models = [
        'meta-llama/Meta-Llama-3.1-70B-Instruct',
        'meta-llama/Meta-Llama-3.1-405B-Instruct-FP8',
        'CohereForAI/c4ai-command-r-plus',
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO',
        '01-ai/Yi-1.5-34B-Chat',
        'mistralai/Mistral-7B-Instruct-v0.2',
        'microsoft/Phi-3-mini-4k-instruct',
    ]
    
    model_aliases = {
        "mistralai/Mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.2"
    }

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        **kwargs) -> CreateResult:
        
        if (model in cls.models) :
            
            session = cf_reqs.Session()
            session.headers = {
                'accept': '*/*',
                'accept-language': 'en',
                'cache-control': 'no-cache',
                'origin': 'https://huggingface.co',
                'pragma': 'no-cache',
                'priority': 'u=1, i',
                'referer': 'https://huggingface.co/chat/',
                'sec-ch-ua': '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"macOS"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
            }

            print(model)
            json_data = {
                'model': model,
            }

            response = session.post('https://huggingface.co/chat/conversation', json=json_data)
            conversationId = response.json()['conversationId']

            response = session.get(f'https://huggingface.co/chat/conversation/{conversationId}/__data.json?x-sveltekit-invalidated=01',)

            data: list = (response.json())["nodes"][1]["data"]
            keys: list[int] = data[data[0]["messages"]]
            message_keys: dict = data[keys[0]]
            messageId: str = data[message_keys["id"]]

            settings = {
                "inputs":format_prompt(messages),
                "id":messageId,
                "is_retry":False,
                "is_continue":False,
                "web_search":False,
                "tools":[]
            }

            headers = {
                'accept': '*/*',
                'accept-language': 'en',
                'cache-control': 'no-cache',
                'origin': 'https://huggingface.co',
                'pragma': 'no-cache',
                'priority': 'u=1, i',
                'referer': f'https://huggingface.co/chat/conversation/{conversationId}',
                'sec-ch-ua': '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"macOS"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
            }


            files = {
                'data': (None,  json.dumps(settings, separators=(',', ':'))),
            }

            response = requests.post(f'https://huggingface.co/chat/conversation/{conversationId}',
                cookies=session.cookies,
                headers=headers,
                files=files,
            )

            first_token = True
            for line in response.iter_lines():
                line = json.loads(line)
                
                if "type" not in line:
                    raise RuntimeError(f"Response: {line}")
                
                elif line["type"] == "stream":
                    token = line["token"]
                    if first_token:
                        token = token.lstrip().replace('\u0000', '')
                        first_token = False
                    
                    else:
                        token = token.replace('\u0000', '')

                    yield (token)
                
                elif line["type"] == "finalAnswer":
                    break