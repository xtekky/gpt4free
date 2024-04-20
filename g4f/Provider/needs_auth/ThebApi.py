from __future__ import annotations

from ...typing import CreateResult, Messages
from .Openai import Openai

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

class ThebApi(Openai):
    label = "TheB.AI API"
    url = "https://theb.ai"
    working = True
    needs_auth = True
    default_model = "gpt-3.5-turbo"
    models = list(models)

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_base: str = "https://api.theb.ai/v1",
        temperature: float = 1,
        top_p: float = 1,
        **kwargs
    ) -> CreateResult:
        if "auth" in kwargs:
            kwargs["api_key"] = kwargs["auth"]
        system_message = "\n".join([message["content"] for message in messages if message["role"] == "system"])
        if not system_message:
            system_message = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture."
        messages = [message for message in messages if message["role"] != "system"]
        data = {
            "model_params": {
                "system_prompt": system_message,
                "temperature": temperature,
                "top_p": top_p,
            }
        }
        return super().create_async_generator(model, messages, api_base=api_base, extra_data=data, **kwargs)