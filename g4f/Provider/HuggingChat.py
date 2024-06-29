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
        "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
        'CohereForAI/c4ai-command-r-plus',
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'google/gemma-1.1-7b-it',
        'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO',
        'mistralai/Mistral-7B-Instruct-v0.2',
        'meta-llama/Meta-Llama-3-70B-Instruct',
        'microsoft/Phi-3-mini-4k-instruct',
        '01-ai/Yi-1.5-34B-Chat'
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
        **kwargs
    ) -> CreateResult:
        
        if (model in cls.models) :
            
            session = requests.Session()
            headers = {
                'accept'            : '*/*',
                'accept-language'   : 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
                'cache-control'     : 'no-cache',
                'origin'            : 'https://huggingface.co',
                'pragma'            : 'no-cache',
                'priority'          : 'u=1, i',
                'referer'           : 'https://huggingface.co/chat/',
                'sec-ch-ua'         : '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
                'sec-ch-ua-mobile'  : '?0',
                'sec-ch-ua-platform': '"macOS"',
                'sec-fetch-dest'    : 'empty',
                'sec-fetch-mode'    : 'cors',
                'sec-fetch-site'    : 'same-origin',
                'user-agent'        : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
            }

            json_data = {
                'searchEnabled'     : True,
                'activeModel'       : 'CohereForAI/c4ai-command-r-plus', # doesn't matter
                'hideEmojiOnSidebar': False,
                'customPrompts'     : {},
                'assistants'        : [],
                'tools'             : {},
                'disableStream'     : False,
                'recentlySaved'     : False,
                'ethicsModalAccepted'   : True,
                'ethicsModalAcceptedAt' : None,
                'shareConversationsWithModelAuthors': False,
            }

            response = cf_reqs.post('https://huggingface.co/chat/settings', headers=headers, json=json_data)
            session.cookies.update(response.cookies)

            response = session.post('https://huggingface.co/chat/conversation', 
                                    headers=headers, json={'model': model})

            conversationId = response.json()['conversationId']
            response       = session.get(f'https://huggingface.co/chat/conversation/{conversationId}/__data.json?x-sveltekit-invalidated=11',
                headers=headers,
            )

            messageId = extract_id(response.json())

            settings = {
                "inputs"      : format_prompt(messages),
                "id"          : messageId,
                "is_retry"    : False,
                "is_continue" : False,
                "web_search"  : False,
                
                # TODO // add feature to enable/disable tools
                "tools": {
                    "websearch"         : True,
                    "document_parser"   : False,
                    "query_calculator"  : False,
                    "image_generation"  : False,
                    "image_editing"     : False,
                    "fetch_url"         : False,
                }
            }

            payload = {
                "data": json.dumps(settings),
            }

            response = session.post(f"https://huggingface.co/chat/conversation/{conversationId}",
                headers=headers, data=payload, stream=True,
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

                    yield token
                
                elif line["type"] == "finalAnswer":
                    break

def extract_id(response: dict) -> str:
    data = response["nodes"][1]["data"]
    uuid_pattern = re.compile(
        r"^[\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12}$"
    )
    for item in data:
        if type(item) == str and uuid_pattern.match(item):
            return item

    return None
