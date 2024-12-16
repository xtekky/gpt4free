from __future__ import annotations

from ..typing import AsyncResult, Messages
from .needs_auth import OpenaiAPI

class DeepInfraChat(OpenaiAPI):
    label = "DeepInfra Chat"
    url = "https://deepinfra.com/chat"
    working = True
    api_base = "https://api.deepinfra.com/v1/openai"

    default_model = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
    models = [
        'meta-llama/Meta-Llama-3.1-8B-Instruct',
        default_model,
        'Qwen/QwQ-32B-Preview',
        'microsoft/WizardLM-2-8x22B',
        'Qwen/Qwen2.5-72B-Instruct',
        'Qwen/Qwen2.5-Coder-32B-Instruct',
        'nvidia/Llama-3.1-Nemotron-70B-Instruct',
    ]
    model_aliases = {
        "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "qwq-32b": "Qwen/QwQ-32B-Preview",
        "wizardlm-2-8x22b": "microsoft/WizardLM-2-8x22B",
        "qwen-2-72b": "Qwen/Qwen2.5-72B-Instruct",
        "qwen-2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "nemotron-70b": "nvidia/Llama-3.1-Nemotron-70B-Instruct",
    }

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            'Accept-Language': 'en-US,en;q=0.9',
            'Content-Type': 'application/json',
            'Origin': 'https://deepinfra.com',
            'Referer': 'https://deepinfra.com/',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'X-Deepinfra-Source': 'web-page',
            'accept': 'text/event-stream',
        }
        return super().create_async_generator(model, messages, proxy, headers=headers, **kwargs)