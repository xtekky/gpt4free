import base64, json, uuid, quickjs

from curl_cffi      import requests
from ..typing       import Any, CreateResult, TypedDict
from .base_provider import BaseProvider


class Vercel(BaseProvider):
    url                   = "https://play.vercel.ai"
    working               = True
    supports_gpt_35_turbo = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:
        
        if model in ["gpt-3.5-turbo", "gpt-4"]:
            model = "openai:" + model
        yield _chat(model_id=model, messages=messages)


def _chat(model_id: str, messages: list[dict[str, str]]) -> str:
    session = requests.Session(impersonate="chrome107")

    url     = "https://sdk.vercel.ai/api/generate"
    header  = _create_header(session)
    payload = _create_payload(model_id, messages)

    response = session.post(url=url, headers=header, json=payload)
    response.raise_for_status()
    return response.text


def _create_payload(model_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    default_params = model_info[model_id]["default_params"]
    return {
        "messages": messages,
        "playgroundId": str(uuid.uuid4()),
        "chatIndex": 0,
        "model": model_id} | default_params


def _create_header(session: requests.Session):
    custom_encoding = _get_custom_encoding(session)
    return {"custom-encoding": custom_encoding}

# based on https://github.com/ading2210/vercel-llm-api
def _get_custom_encoding(session: requests.Session):
    url = "https://sdk.vercel.ai/openai.jpeg"
    response = session.get(url=url)

    data = json.loads(base64.b64decode(response.text, validate=True))
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
