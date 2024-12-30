from __future__ import annotations

import json

try:
    from curl_cffi.requests import Session, CurlMime
    has_curl_cffi = True
except ImportError:
    has_curl_cffi = False

from ..base_provider import ProviderModelMixin, AbstractProvider
from ..helper import format_prompt
from ...typing import CreateResult, Messages, Cookies
from ...errors import MissingRequirementsError
from ...requests.raise_for_status import raise_for_status
from ...providers.response import JsonConversation, ImageResponse, Sources
from ...cookies import get_cookies
from ... import debug

class Conversation(JsonConversation):
    def __init__(self, conversation_id: str, message_id: str):
        self.conversation_id: str = conversation_id
        self.message_id: str = message_id

class HuggingChat(AbstractProvider, ProviderModelMixin):
    url = "https://huggingface.co/chat"
    
    working = True
    supports_stream = True
    needs_auth = True
    
    default_model = "Qwen/Qwen2.5-72B-Instruct"
    default_image_model = "black-forest-labs/FLUX.1-dev"
    image_models = [    
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-schnell",
    ]
    models = [
        default_model,
        'meta-llama/Llama-3.3-70B-Instruct',
        'CohereForAI/c4ai-command-r-plus-08-2024',
        'Qwen/QwQ-32B-Preview',
        'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
        'Qwen/Qwen2.5-Coder-32B-Instruct',
        'meta-llama/Llama-3.2-11B-Vision-Instruct',
        'NousResearch/Hermes-3-Llama-3.1-8B',
        'mistralai/Mistral-Nemo-Instruct-2407',
        'microsoft/Phi-3.5-mini-instruct',
        *image_models
    ]
    model_aliases = {
        ### Chat ###
        "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "command-r-plus": "CohereForAI/c4ai-command-r-plus-08-2024",
        "qwq-32b": "Qwen/QwQ-32B-Preview",
        "nemotron-70b": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "qwen-2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "llama-3.2-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "hermes-3": "NousResearch/Hermes-3-Llama-3.1-8B",
        "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
        "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",

        ### Image ###
        "flux-dev": "black-forest-labs/FLUX.1-dev",
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    }

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        return_conversation: bool = False,
        conversation: Conversation = None,
        web_search: bool = False,
        cookies: Cookies = None,
        **kwargs
    ) -> CreateResult:
        if not has_curl_cffi:
            raise MissingRequirementsError('Install "curl_cffi" package | pip install -U curl_cffi')
        model = cls.get_model(model)
        if cookies is None:
            cookies = get_cookies("huggingface.co")

        session = Session(cookies=cookies)
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

        if conversation is None:
            conversationId = cls.create_conversation(session, model)
            messageId = cls.fetch_message_id(session, conversationId)
            conversation = Conversation(conversationId, messageId)
            if return_conversation:
                yield conversation
            inputs = format_prompt(messages)
        else:
            conversation.message_id = cls.fetch_message_id(session, conversation.conversation_id)
            inputs = messages[-1]["content"]

        debug.log(f"Use conversation: {conversation.conversation_id} Use message: {conversation.message_id}")

        settings = {
            "inputs": inputs,
            "id": conversation.message_id,
            "is_retry": False,
            "is_continue": False,
            "web_search": web_search,
            "tools": ["000000000000000000000001"] if model in cls.image_models else [],
        }

        headers = {
            'accept': '*/*',
            'accept-language': 'en',
            'cache-control': 'no-cache',
            'origin': 'https://huggingface.co',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': f'https://huggingface.co/chat/conversation/{conversation.conversation_id}',
            'sec-ch-ua': '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
        }

        data = CurlMime()
        data.addpart('data', data=json.dumps(settings, separators=(',', ':')))

        response = session.post(
            f'https://huggingface.co/chat/conversation/{conversation.conversation_id}',
            cookies=session.cookies,
            headers=headers,
            multipart=data,
            stream=True
        )
        raise_for_status(response)

        full_response = ""
        sources = None
        for line in response.iter_lines():
            if not line:
                continue
            try:
                line = json.loads(line)
            except json.JSONDecodeError as e:
                debug.log(f"Failed to decode JSON: {line}, error: {e}")
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
            elif line["type"] == "file":
                url = f"https://huggingface.co/chat/conversation/{conversation.conversation_id}/output/{line['sha']}"
                yield ImageResponse(url, alt=messages[-1]["content"], options={"cookies": cookies})
            elif line["type"] == "webSearch" and "sources" in line:
                sources = Sources(line["sources"])

        full_response = full_response.replace('<|im_end|', '').strip()
        if not stream:
            yield full_response
        if sources is not None:
            yield sources

    @classmethod
    def create_conversation(cls, session: Session, model: str):
        if model in cls.image_models:
            model = cls.default_model
        json_data = {
            'model': model,
        }
        response = session.post('https://huggingface.co/chat/conversation', json=json_data)
        raise_for_status(response)

        return response.json().get('conversationId')

    @classmethod
    def fetch_message_id(cls, session: Session, conversation_id: str):
        # Get the data response and parse it properly
        response = session.get(f'https://huggingface.co/chat/conversation/{conversation_id}/__data.json?x-sveltekit-invalidated=11')
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

            data = json_data["nodes"][1]["data"]
            keys = data[data[0]["messages"]]
            message_keys = data[keys[-1]]
            return data[message_keys["id"]]

        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Failed to extract message ID: {str(e)}")
