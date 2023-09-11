from __future__ import annotations

import base64, json, uuid, quickjs, random
from curl_cffi.requests import AsyncSession

from ..typing       import Any, TypedDict
from .base_provider import AsyncProvider


class Vercel(AsyncProvider):
    url                   = "https://sdk.vercel.ai"
    working               = True
    supports_gpt_35_turbo = True
    model                 = "replicate:replicate/llama-2-70b-chat"

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        **kwargs
    ) -> str:
        if model in ["gpt-3.5-turbo", "gpt-4"]:
            model = "openai:" + model
        model = model if model else cls.model
        proxies = None
        if proxy:
            if "://" not in proxy:
                proxy = "http://" + proxy
            proxies = {"http": proxy, "https": proxy}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.{rand1}.{rand2} Safari/537.36".format(
                rand1=random.randint(0,9999),
                rand2=random.randint(0,9999)
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.5",
            "TE": "trailers",
        }
        async with AsyncSession(headers=headers, proxies=proxies, impersonate="chrome107") as session:
            response = await session.get(cls.url + "/openai.jpeg")
            response.raise_for_status()
            custom_encoding = _get_custom_encoding(response.text)
            headers = {
                "Content-Type": "application/json",
                "Custom-Encoding": custom_encoding,
            }
            data = _create_payload(model, messages)
            response = await session.post(cls.url + "/api/generate", json=data, headers=headers)
            response.raise_for_status()
            return response.text


def _create_payload(model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    if model not in model_info:
        raise ValueError(f'Model are not supported: {model}')
    default_params = model_info[model]["default_params"]
    return {
        "messages": messages,
        "playgroundId": str(uuid.uuid4()),
        "chatIndex": 0,
        "model": model
    } | default_params

# based on https://github.com/ading2210/vercel-llm-api
def _get_custom_encoding(text: str) -> str:
    data = json.loads(base64.b64decode(text, validate=True))
    script = """
      String.prototype.fontcolor = function() {{
        return `<font>${{this}}</font>`
      }}
      var globalThis = {{marker: "mark"}};
      ({script})({key})
    """.format(
        script=data["c"], key=data["a"]
    )
    context = quickjs.Context()  # type: ignore
    token_data = json.loads(context.eval(script).json())  # type: ignore
    token_data[2] = "mark"
    token = {"r": token_data, "t": data["t"]}
    token_str = json.dumps(token, separators=(",", ":")).encode("utf-16le")
    return base64.b64encode(token_str).decode()


class ModelInfo(TypedDict):
    id: str
    default_params: dict[str, Any]


model_info: dict[str, ModelInfo] = {
    "anthropic:claude-instant-v1": {
        "id": "anthropic:claude-instant-v1",
        "default_params": {
            "temperature": 1,
            "maxTokens": 200,
            "topP": 1,
            "topK": 1,
            "presencePenalty": 1,
            "frequencyPenalty": 1,
            "stopSequences": ["\n\nHuman:"],
        },
    },
    "anthropic:claude-v1": {
        "id": "anthropic:claude-v1",
        "default_params": {
            "temperature": 1,
            "maxTokens": 200,
            "topP": 1,
            "topK": 1,
            "presencePenalty": 1,
            "frequencyPenalty": 1,
            "stopSequences": ["\n\nHuman:"],
        },
    },
    "anthropic:claude-v2": {
        "id": "anthropic:claude-v2",
        "default_params": {
            "temperature": 1,
            "maxTokens": 200,
            "topP": 1,
            "topK": 1,
            "presencePenalty": 1,
            "frequencyPenalty": 1,
            "stopSequences": ["\n\nHuman:"],
        },
    },
    "replicate:a16z-infra/llama7b-v2-chat": {
        "id": "replicate:a16z-infra/llama7b-v2-chat",
        "default_params": {
            "temperature": 0.75,
            "maxTokens": 500,
            "topP": 1,
            "repetitionPenalty": 1,
        },
    },
    "replicate:a16z-infra/llama13b-v2-chat": {
        "id": "replicate:a16z-infra/llama13b-v2-chat",
        "default_params": {
            "temperature": 0.75,
            "maxTokens": 500,
            "topP": 1,
            "repetitionPenalty": 1,
        },
    },
    "replicate:replicate/llama-2-70b-chat": {
        "id": "replicate:replicate/llama-2-70b-chat",
        "default_params": {
            "temperature": 0.75,
            "maxTokens": 1000,
            "topP": 1,
            "repetitionPenalty": 1,
        },
    },
    "huggingface:bigscience/bloom": {
        "id": "huggingface:bigscience/bloom",
        "default_params": {
            "temperature": 0.5,
            "maxTokens": 200,
            "topP": 0.95,
            "topK": 4,
            "repetitionPenalty": 1.03,
        },
    },
    "huggingface:google/flan-t5-xxl": {
        "id": "huggingface:google/flan-t5-xxl",
        "default_params": {
            "temperature": 0.5,
            "maxTokens": 200,
            "topP": 0.95,
            "topK": 4,
            "repetitionPenalty": 1.03,
        },
    },
    "huggingface:EleutherAI/gpt-neox-20b": {
        "id": "huggingface:EleutherAI/gpt-neox-20b",
        "default_params": {
            "temperature": 0.5,
            "maxTokens": 200,
            "topP": 0.95,
            "topK": 4,
            "repetitionPenalty": 1.03,
            "stopSequences": [],
        },
    },
    "huggingface:OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5": {
        "id": "huggingface:OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
        "default_params": {"maxTokens": 200, "typicalP": 0.2, "repetitionPenalty": 1},
    },
    "huggingface:OpenAssistant/oasst-sft-1-pythia-12b": {
        "id": "huggingface:OpenAssistant/oasst-sft-1-pythia-12b",
        "default_params": {"maxTokens": 200, "typicalP": 0.2, "repetitionPenalty": 1},
    },
    "huggingface:bigcode/santacoder": {
        "id": "huggingface:bigcode/santacoder",
        "default_params": {
            "temperature": 0.5,
            "maxTokens": 200,
            "topP": 0.95,
            "topK": 4,
            "repetitionPenalty": 1.03,
        },
    },
    "cohere:command-light-nightly": {
        "id": "cohere:command-light-nightly",
        "default_params": {
            "temperature": 0.9,
            "maxTokens": 200,
            "topP": 1,
            "topK": 0,
            "presencePenalty": 0,
            "frequencyPenalty": 0,
            "stopSequences": [],
        },
    },
    "cohere:command-nightly": {
        "id": "cohere:command-nightly",
        "default_params": {
            "temperature": 0.9,
            "maxTokens": 200,
            "topP": 1,
            "topK": 0,
            "presencePenalty": 0,
            "frequencyPenalty": 0,
            "stopSequences": [],
        },
    },
    "openai:gpt-4": {
        "id": "openai:gpt-4",
        "default_params": {
            "temperature": 0.7,
            "maxTokens": 500,
            "topP": 1,
            "presencePenalty": 0,
            "frequencyPenalty": 0,
            "stopSequences": [],
        },
    },
    "openai:gpt-4-0613": {
        "id": "openai:gpt-4-0613",
        "default_params": {
            "temperature": 0.7,
            "maxTokens": 500,
            "topP": 1,
            "presencePenalty": 0,
            "frequencyPenalty": 0,
            "stopSequences": [],
        },
    },
    "openai:code-davinci-002": {
        "id": "openai:code-davinci-002",
        "default_params": {
            "temperature": 0.5,
            "maxTokens": 200,
            "topP": 1,
            "presencePenalty": 0,
            "frequencyPenalty": 0,
            "stopSequences": [],
        },
    },
    "openai:gpt-3.5-turbo": {
        "id": "openai:gpt-3.5-turbo",
        "default_params": {
            "temperature": 0.7,
            "maxTokens": 500,
            "topP": 1,
            "topK": 1,
            "presencePenalty": 1,
            "frequencyPenalty": 1,
            "stopSequences": [],
        },
    },
    "openai:gpt-3.5-turbo-16k": {
        "id": "openai:gpt-3.5-turbo-16k",
        "default_params": {
            "temperature": 0.7,
            "maxTokens": 500,
            "topP": 1,
            "topK": 1,
            "presencePenalty": 1,
            "frequencyPenalty": 1,
            "stopSequences": [],
        },
    },
    "openai:gpt-3.5-turbo-16k-0613": {
        "id": "openai:gpt-3.5-turbo-16k-0613",
        "default_params": {
            "temperature": 0.7,
            "maxTokens": 500,
            "topP": 1,
            "topK": 1,
            "presencePenalty": 1,
            "frequencyPenalty": 1,
            "stopSequences": [],
        },
    },
    "openai:text-ada-001": {
        "id": "openai:text-ada-001",
        "default_params": {
            "temperature": 0.5,
            "maxTokens": 200,
            "topP": 1,
            "presencePenalty": 0,
            "frequencyPenalty": 0,
            "stopSequences": [],
        },
    },
    "openai:text-babbage-001": {
        "id": "openai:text-babbage-001",
        "default_params": {
            "temperature": 0.5,
            "maxTokens": 200,
            "topP": 1,
            "presencePenalty": 0,
            "frequencyPenalty": 0,
            "stopSequences": [],
        },
    },
    "openai:text-curie-001": {
        "id": "openai:text-curie-001",
        "default_params": {
            "temperature": 0.5,
            "maxTokens": 200,
            "topP": 1,
            "presencePenalty": 0,
            "frequencyPenalty": 0,
            "stopSequences": [],
        },
    },
    "openai:text-davinci-002": {
        "id": "openai:text-davinci-002",
        "default_params": {
            "temperature": 0.5,
            "maxTokens": 200,
            "topP": 1,
            "presencePenalty": 0,
            "frequencyPenalty": 0,
            "stopSequences": [],
        },
    },
    "openai:text-davinci-003": {
        "id": "openai:text-davinci-003",
        "default_params": {
            "temperature": 0.5,
            "maxTokens": 200,
            "topP": 1,
            "presencePenalty": 0,
            "frequencyPenalty": 0,
            "stopSequences": [],
        },
    },
}
