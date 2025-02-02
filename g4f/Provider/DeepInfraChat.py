from __future__ import annotations

from ..typing import AsyncResult, Messages
from .template import OpenaiTemplate

class DeepInfraChat(OpenaiTemplate):
    url = "https://deepinfra.com/chat"
    api_base = "https://api.deepinfra.com/v1/openai"
    working = True

    default_model = 'meta-llama/Llama-3.3-70B-Instruct-Turbo'
    models = [
        'meta-llama/Llama-3.3-70B-Instruct',
        'meta-llama/Meta-Llama-3.1-8B-Instruct',
        default_model,
        'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        'deepseek-ai/DeepSeek-V3',
        'Qwen/QwQ-32B-Preview',
        'microsoft/WizardLM-2-8x22B',
        'microsoft/WizardLM-2-7B',
        'Qwen/Qwen2.5-72B-Instruct',
        'Qwen/Qwen2.5-Coder-32B-Instruct',
        'nvidia/Llama-3.1-Nemotron-70B-Instruct',
    ]
    model_aliases = {
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "deepseek-v3": "deepseek-ai/DeepSeek-V3",
        "qwq-32b": "Qwen/QwQ-32B-Preview",
        "wizardlm-2-8x22b": "microsoft/WizardLM-2-8x22B",
        "wizardlm-2-7b": "microsoft/WizardLM-2-7B",
        "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
        "qwen-2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "nemotron-70b": "nvidia/Llama-3.1-Nemotron-70B-Instruct",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        headers: dict = {},
        **kwargs
    ) -> AsyncResult:
        headers = {
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://deepinfra.com',
            'Referer': 'https://deepinfra.com/',
            'X-Deepinfra-Source': 'web-page',
            **headers
        }
        async for chunk in super().create_async_generator(model, messages, headers=headers, **kwargs):
            yield chunk
