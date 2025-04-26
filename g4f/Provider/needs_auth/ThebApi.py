from __future__ import annotations

from ...typing import CreateResult, Messages
from ..helper import filter_none
from ..template import OpenaiTemplate

models = {
    "theb-ai": "TheB.AI",
    "gpt-3.5-turbo": "GPT-3.5",
    "gpt-4-turbo": "GPT-4 Turbo",
    "gpt-4": "GPT-4",
    "claude-3.5-sonnet": "Claude",
    "llama-2-7b-chat": "Llama 2 7B",
    "llama-2-13b-chat": "Llama 2 13B",
    "llama-2-70b-chat": "Llama 2 70B",
    "code-llama-7b": "Code Llama 7B",
    "code-llama-13b": "Code Llama 13B",
    "code-llama-34b": "Code Llama 34B",
    "qwen-2-72b": "Qwen"
}

class ThebApi(OpenaiTemplate):
    label = "TheB.AI API"
    url = "https://theb.ai"
    login_url = "https://beta.theb.ai/home"
    api_base = "https://api.theb.ai/v1"
    working = True
    needs_auth = True

    default_model = "theb-ai"
    fallback_models = list(models)

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        temperature: float = None,
        top_p: float = None,
        **kwargs
    ) -> CreateResult:
        system_message = "\n".join([message["content"] for message in messages if message["role"] == "system"])
        messages = [message for message in messages if message["role"] != "system"]
        data = {
            "model_params": filter_none(
                system_prompt=system_message,
                temperature=temperature,
                top_p=top_p,
            )
        }
        return super().create_async_generator(model, messages, extra_data=data, **kwargs)
