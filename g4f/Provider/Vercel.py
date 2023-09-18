from __future__ import annotations

import base64, json, uuid, quickjs, random
from curl_cffi.requests import AsyncSession

from ..typing       import Any, TypedDict
from .base_provider import AsyncProvider


class Vercel(AsyncProvider):
    url                   = "https://sdk.vercel.ai"
    working               = False
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
        return

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
