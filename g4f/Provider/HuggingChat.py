from __future__ import annotations

import json
import requests

try:
    from curl_cffi.requests import Session
    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False
from ..typing import CreateResult, Messages
from ..errors import MissingRequirementsError
from ..requests.raise_for_status import raise_for_status
from .base_provider import ProviderModelMixin, AbstractProvider
from .helper import format_prompt

class HuggingChat(AbstractProvider, ProviderModelMixin):
    url = "https://huggingface.co/chat"
    working = True
    supports_stream = True
    default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    models = [
        'meta-llama/Meta-Llama-3.1-70B-Instruct',
        'CohereForAI/c4ai-command-r-plus-08-2024',
        'Qwen/Qwen2.5-72B-Instruct',
        'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
        'Qwen/Qwen2.5-Coder-32B-Instruct',
        'meta-llama/Llama-3.2-11B-Vision-Instruct',
        'NousResearch/Hermes-3-Llama-3.1-8B',
        'mistralai/Mistral-Nemo-Instruct-2407',
        'microsoft/Phi-3.5-mini-instruct',
    ]

    model_aliases = {
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "command-r-plus": "CohereForAI/c4ai-command-r-plus-08-2024",
        "qwen-2-72b": "Qwen/Qwen2.5-72B-Instruct",
        "nemotron-70b": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "qwen-2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "llama-3.2-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "hermes-3": "NousResearch/Hermes-3-Llama-3.1-8B",
        "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
        "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    }

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        **kwargs
    ) -> CreateResult:
        if not has_curl_cffi:
            raise MissingRequirementsError('Install "curl_cffi" package | pip install -U curl_cffi')
        model = cls.get_model(model)
        
        if model in cls.models:
            session = Session()
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
            json_data = {
                'model': model,
            }
            response = session.post('https://huggingface.co/chat/conversation', json=json_data)
            raise_for_status(response)

            conversationId = response.json().get('conversationId')
            
            # Get the data response and parse it properly
            response = session.get(f'https://huggingface.co/chat/conversation/{conversationId}/__data.json?x-sveltekit-invalidated=11')
            raise_for_status(response)

            # Split the response content by newlines and parse each line as JSON
            try:
                json_data = None
                for line in response.text.split('\n'):
                    if line.strip():
                        try:
                            parsed = json.loads(line)
                            if isinstance(parsed, dict) and "nodes" in parsed:
                                json_data = parsed
                                break
                        except json.JSONDecodeError:
                            continue
                            
                if not json_data:
                    raise RuntimeError("Failed to parse response data")

                data: list = json_data["nodes"][1]["data"]
                keys: list[int] = data[data[0]["messages"]]
                message_keys: dict = data[keys[0]]
                messageId: str = data[message_keys["id"]]

            except (KeyError, IndexError, TypeError) as e:
                raise RuntimeError(f"Failed to extract message ID: {str(e)}")

            settings = {
                "inputs": format_prompt(messages),
                "id": messageId,
                "is_retry": False,
                "is_continue": False,
                "web_search": False,
                "tools": []
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
                'data': (None, json.dumps(settings, separators=(',', ':'))),
            }

            response = requests.post(
                f'https://huggingface.co/chat/conversation/{conversationId}',
                cookies=session.cookies,
                headers=headers,
                files=files,
            )
            raise_for_status(response)

            full_response = ""
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    line = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON: {line}, error: {e}")
                    continue
                
                if "type" not in line:
                    raise RuntimeError(f"Response: {line}")
                
                elif line["type"] == "stream":
                    token = line["token"].replace('\u0000', '')
                    full_response += token
                    if stream:
                        yield token
                
                elif line["type"] == "finalAnswer":
                    break
            
            full_response = full_response.replace('<|im_end|', '').replace('\u0000', '').strip()

            if not stream:
                yield full_response