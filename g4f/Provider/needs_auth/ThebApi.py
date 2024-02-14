from __future__ import annotations

import requests

from ...typing import Any, CreateResult, Messages
from ..base_provider import AbstractProvider, ProviderModelMixin
from ...errors import MissingAuthError

models = {
    "theb-ai": "TheB.AI",
    "gpt-3.5-turbo": "GPT-3.5",
    "gpt-3.5-turbo-16k": "GPT-3.5-16K",
    "gpt-4-turbo": "GPT-4 Turbo",
    "gpt-4": "GPT-4",
    "gpt-4-32k": "GPT-4 32K",
    "claude-2": "Claude 2",
    "claude-1": "Claude",
    "claude-1-100k": "Claude 100K",
    "claude-instant-1": "Claude Instant",
    "claude-instant-1-100k": "Claude Instant 100K",
    "palm-2": "PaLM 2",
    "palm-2-codey": "Codey",
    "vicuna-13b-v1.5": "Vicuna v1.5 13B",
    "llama-2-7b-chat": "Llama 2 7B",
    "llama-2-13b-chat": "Llama 2 13B",
    "llama-2-70b-chat": "Llama 2 70B",
    "code-llama-7b": "Code Llama 7B",
    "code-llama-13b": "Code Llama 13B",
    "code-llama-34b": "Code Llama 34B",
    "qwen-7b-chat": "Qwen 7B"
}

class ThebApi(AbstractProvider, ProviderModelMixin):
    url = "https://theb.ai"
    working = True
    needs_auth = True
    default_model = "gpt-3.5-turbo"
    models = list(models)

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        auth: str = None,
        proxy: str = None,
        **kwargs
    ) -> CreateResult:
        if not auth:
            raise MissingAuthError("Missing auth")
        headers = {
            'accept': 'application/json',
            'authorization': f'Bearer {auth}',
            'content-type': 'application/json',
        }
        # response = requests.get("https://api.baizhi.ai/v1/models", headers=headers).json()["data"]
        # models = dict([(m["id"], m["name"]) for m in response])
        # print(json.dumps(models, indent=4))
        data: dict[str, Any] = {
            "model": cls.get_model(model),
            "messages": messages,
            "stream": False,
            "model_params": {
                "system_prompt": kwargs.get("system_message", "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture."),
                "temperature": 1,
                "top_p": 1,
                **kwargs
            }
        }
        response = requests.post(
            "https://api.theb.ai/v1/chat/completions",
            headers=headers,
            json=data,
            proxies={"https": proxy}
        )
        try:
            response.raise_for_status()
            yield response.json()["choices"][0]["message"]["content"]
        except:
            raise RuntimeError(f"Response: {next(response.iter_lines()).decode()}")