from __future__ import annotations

from ..typing import AsyncResult, Messages
from .template import OpenaiTemplate

class DeepInfraChat(OpenaiTemplate):
    url = "https://deepinfra.com/chat"
    api_base = "https://api.deepinfra.com/v1/openai"
    working = True

    default_model = 'meta-llama/Llama-3.3-70B-Instruct-Turbo'
    models = [
        'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'meta-llama/Llama-3.2-90B-Vision-Instruct',
        default_model,
        'deepseek-ai/DeepSeek-V3',
        'mistralai/Mistral-Small-24B-Instruct-2501',
        'deepseek-ai/DeepSeek-R1',
        'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'microsoft/phi-4',
        'microsoft/WizardLM-2-8x22B',
        'Qwen/Qwen2.5-72B-Instruct',
    ]
    model_aliases = {
        "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.2-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "deepseek-v3": "deepseek-ai/DeepSeek-V3",
        "mixtral-small-28b": "mistralai/Mistral-Small-24B-Instruct-2501",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "phi-4": "microsoft/phi-4",
        "wizardlm-2-8x22b": "microsoft/WizardLM-2-8x22B",
        "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        top_p: float = 0.9,
        temperature: float = 0.7,
        max_tokens: int = None,
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
