from __future__ import annotations

import json, base64, requests, execjs, random, uuid

from ..typing       import Messages, TypedDict, CreateResult
from .base_provider import BaseProvider
from abc            import abstractmethod


class Vercel(BaseProvider):
    url = 'https://sdk.vercel.ai'
    working               = True
    supports_gpt_35_turbo = True
    supports_stream       = True

    @staticmethod
    @abstractmethod
    def create_completion(
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        **kwargs
    ) -> CreateResult:
        if not model:
            model = "gpt-3.5-turbo"
        elif model not in model_info:
            raise ValueError(f"Model are not supported: {model}")

        headers = {
            'authority'         : 'sdk.vercel.ai',
            'accept'            : '*/*',
            'accept-language'   : 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'cache-control'     : 'no-cache',
            'content-type'      : 'application/json',
            'custom-encoding'   : get_anti_bot_token(),
            'origin'            : 'https://sdk.vercel.ai',
            'pragma'            : 'no-cache',
            'referer'           : 'https://sdk.vercel.ai/',
            'sec-ch-ua'         : '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
            'sec-ch-ua-mobile'  : '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest'    : 'empty',
            'sec-fetch-mode'    : 'cors',
            'sec-fetch-site'    : 'same-origin',
            'user-agent'        :  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.%s.%s Safari/537.36' % (
                random.randint(99, 999),
                random.randint(99, 999)
            )
        }

        json_data = {
            'model'       : model_info[model]['id'],
            'messages'    : messages,
            'playgroundId': str(uuid.uuid4()),
            'chatIndex'   : 0,
            **model_info[model]['default_params'],
            **kwargs
        }

        max_retries  = kwargs.get('max_retries', 20)
        for i in range(max_retries):
            response = requests.post('https://sdk.vercel.ai/api/generate', 
                                    headers=headers, json=json_data, stream=True, proxies={"https": proxy})
            try:
                response.raise_for_status()
            except Exception:
                continue
            for token in response.iter_content(chunk_size=None):
                yield token.decode()
            break


def get_anti_bot_token() -> str:
    headers = {
        'authority'         : 'sdk.vercel.ai',
        'accept'            : '*/*',
        'accept-language'   : 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
        'cache-control'     : 'no-cache',
        'pragma'            : 'no-cache',
        'referer'           : 'https://sdk.vercel.ai/',
        'sec-ch-ua'         : '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
        'sec-ch-ua-mobile'  : '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest'    : 'empty',
        'sec-fetch-mode'    : 'cors',
        'sec-fetch-site'    : 'same-origin',
        'user-agent'        : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.%s.%s Safari/537.36' % (
            random.randint(99, 999), 
            random.randint(99, 999)
        )
    }

    response = requests.get('https://sdk.vercel.ai/openai.jpeg', 
                            headers=headers).text

    raw_data = json.loads(base64.b64decode(response, 
                                    validate=True))

    js_script = '''const globalThis={marker:"mark"};String.prototype.fontcolor=function(){return `<font>${this}</font>`};
        return (%s)(%s)''' % (raw_data['c'], raw_data['a'])

    raw_token = json.dumps({'r': execjs.compile(js_script).call(''), 't': raw_data['t']}, 
                        separators = (",", ":"))

    return base64.b64encode(raw_token.encode('utf-16le')).decode()

class ModelInfo(TypedDict):
    id: str
    default_params: dict[str, Any]

model_info: dict[str, ModelInfo] = {
    'claude-instant-v1': {
        'id': 'anthropic:claude-instant-v1',
        'default_params': {
            'temperature': 1,
            'maximumLength': 1024,
            'topP': 1,
            'topK': 1,
            'presencePenalty': 1,
            'frequencyPenalty': 1,
            'stopSequences': ['\n\nHuman:'],
        },
    },
    'claude-v1': {
        'id': 'anthropic:claude-v1',
        'default_params': {
            'temperature': 1,
            'maximumLength': 1024,
            'topP': 1,
            'topK': 1,
            'presencePenalty': 1,
            'frequencyPenalty': 1,
            'stopSequences': ['\n\nHuman:'],
        },
    },
    'claude-v2': {
        'id': 'anthropic:claude-v2',
        'default_params': {
            'temperature': 1,
            'maximumLength': 1024,
            'topP': 1,
            'topK': 1,
            'presencePenalty': 1,
            'frequencyPenalty': 1,
            'stopSequences': ['\n\nHuman:'],
        },
    },
    'a16z-infra/llama7b-v2-chat': {
        'id': 'replicate:a16z-infra/llama7b-v2-chat',
        'default_params': {
            'temperature': 0.75,
            'maximumLength': 3000,
            'topP': 1,
            'repetitionPenalty': 1,
        },
    },
    'a16z-infra/llama13b-v2-chat': {
        'id': 'replicate:a16z-infra/llama13b-v2-chat',
        'default_params': {
            'temperature': 0.75,
            'maximumLength': 3000,
            'topP': 1,
            'repetitionPenalty': 1,
        },
    },
    'replicate/llama-2-70b-chat': {
        'id': 'replicate:replicate/llama-2-70b-chat',
        'default_params': {
            'temperature': 0.75,
            'maximumLength': 3000,
            'topP': 1,
            'repetitionPenalty': 1,
        },
    },
    'bigscience/bloom': {
        'id': 'huggingface:bigscience/bloom',
        'default_params': {
            'temperature': 0.5,
            'maximumLength': 1024,
            'topP': 0.95,
            'topK': 4,
            'repetitionPenalty': 1.03,
        },
    },
    'google/flan-t5-xxl': {
        'id': 'huggingface:google/flan-t5-xxl',
        'default_params': {
            'temperature': 0.5,
            'maximumLength': 1024,
            'topP': 0.95,
            'topK': 4,
            'repetitionPenalty': 1.03,
        },
    },
    'EleutherAI/gpt-neox-20b': {
        'id': 'huggingface:EleutherAI/gpt-neox-20b',
        'default_params': {
            'temperature': 0.5,
            'maximumLength': 1024,
            'topP': 0.95,
            'topK': 4,
            'repetitionPenalty': 1.03,
            'stopSequences': [],
        },
    },
    'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5': {
        'id': 'huggingface:OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
        'default_params': {
            'maximumLength': 1024,
            'typicalP': 0.2,
            'repetitionPenalty': 1,
        },
    },
    'OpenAssistant/oasst-sft-1-pythia-12b': {
        'id': 'huggingface:OpenAssistant/oasst-sft-1-pythia-12b',
        'default_params': {
            'maximumLength': 1024,
            'typicalP': 0.2,
            'repetitionPenalty': 1,
        },
    },
    'bigcode/santacoder': {
        'id': 'huggingface:bigcode/santacoder',
        'default_params': {
            'temperature': 0.5,
            'maximumLength': 1024,
            'topP': 0.95,
            'topK': 4,
            'repetitionPenalty': 1.03,
        },
    },
    'command-light-nightly': {
        'id': 'cohere:command-light-nightly',
        'default_params': {
            'temperature': 0.9,
            'maximumLength': 1024,
            'topP': 1,
            'topK': 0,
            'presencePenalty': 0,
            'frequencyPenalty': 0,
            'stopSequences': [],
        },
    },
    'command-nightly': {
        'id': 'cohere:command-nightly',
        'default_params': {
            'temperature': 0.9,
            'maximumLength': 1024,
            'topP': 1,
            'topK': 0,
            'presencePenalty': 0,
            'frequencyPenalty': 0,
            'stopSequences': [],
        },
    },
    'gpt-4': {
        'id': 'openai:gpt-4',
        'default_params': {
            'temperature': 0.7,
            'maximumLength': 8192,
            'topP': 1,
            'presencePenalty': 0,
            'frequencyPenalty': 0,
            'stopSequences': [],
        },
    },
    'gpt-4-0613': {
        'id': 'openai:gpt-4-0613',
        'default_params': {
            'temperature': 0.7,
            'maximumLength': 8192,
            'topP': 1,
            'presencePenalty': 0,
            'frequencyPenalty': 0,
            'stopSequences': [],
        },
    },
    'code-davinci-002': {
        'id': 'openai:code-davinci-002',
        'default_params': {
            'temperature': 0.5,
            'maximumLength': 1024,
            'topP': 1,
            'presencePenalty': 0,
            'frequencyPenalty': 0,
            'stopSequences': [],
        },
    },
    'gpt-3.5-turbo': {
        'id': 'openai:gpt-3.5-turbo',
        'default_params': {
            'temperature': 0.7,
            'maximumLength': 4096,
            'topP': 1,
            'topK': 1,
            'presencePenalty': 1,
            'frequencyPenalty': 1,
            'stopSequences': [],
        },
    },
    'gpt-3.5-turbo-16k': {
        'id': 'openai:gpt-3.5-turbo-16k',
        'default_params': {
            'temperature': 0.7,
            'maximumLength': 16280,
            'topP': 1,
            'topK': 1,
            'presencePenalty': 1,
            'frequencyPenalty': 1,
            'stopSequences': [],
        },
    },
    'gpt-3.5-turbo-16k-0613': {
        'id': 'openai:gpt-3.5-turbo-16k-0613',
        'default_params': {
            'temperature': 0.7,
            'maximumLength': 16280,
            'topP': 1,
            'topK': 1,
            'presencePenalty': 1,
            'frequencyPenalty': 1,
            'stopSequences': [],
        },
    },
    'text-ada-001': {
        'id': 'openai:text-ada-001',
        'default_params': {
            'temperature': 0.5,
            'maximumLength': 1024,
            'topP': 1,
            'presencePenalty': 0,
            'frequencyPenalty': 0,
            'stopSequences': [],
        },
    },
    'text-babbage-001': {
        'id': 'openai:text-babbage-001',
        'default_params': {
            'temperature': 0.5,
            'maximumLength': 1024,
            'topP': 1,
            'presencePenalty': 0,
            'frequencyPenalty': 0,
            'stopSequences': [],
        },
    },
    'text-curie-001': {
        'id': 'openai:text-curie-001',
        'default_params': {
            'temperature': 0.5,
            'maximumLength': 1024,
            'topP': 1,
            'presencePenalty': 0,
            'frequencyPenalty': 0,
            'stopSequences': [],
        },
    },
    'text-davinci-002': {
        'id': 'openai:text-davinci-002',
        'default_params': {
            'temperature': 0.5,
            'maximumLength': 1024,
            'topP': 1,
            'presencePenalty': 0,
            'frequencyPenalty': 0,
            'stopSequences': [],
        },
    },
    'text-davinci-003': {
        'id': 'openai:text-davinci-003',
        'default_params': {
            'temperature': 0.5,
            'maximumLength': 4097,
            'topP': 1,
            'presencePenalty': 0,
            'frequencyPenalty': 0,
            'stopSequences': [],
        },
    },
}