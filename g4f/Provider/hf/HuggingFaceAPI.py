from __future__ import annotations

from ..template.OpenaiTemplate import OpenaiTemplate
from .models import model_aliases
from ...providers.types import Messages
from .HuggingChat import HuggingChat
from ... import debug

class HuggingFaceAPI(OpenaiTemplate):
    label = "HuggingFace (Inference API)"
    parent = "HuggingFace"
    url = "https://api-inference.huggingface.com"
    api_base = "https://api-inference.huggingface.co/v1"
    working = True
    needs_auth = True

    default_model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    default_vision_model = default_model
    vision_models = [default_vision_model, "Qwen/Qwen2-VL-7B-Instruct"]
    model_aliases = model_aliases

    @classmethod
    def get_models(cls, **kwargs):
        if not cls.models:
            HuggingChat.get_models()
            cls.models = HuggingChat.text_models.copy()
            for model in cls.vision_models:
                if model not in cls.models:
                    cls.models.append(model)
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_base: str = None,
        max_tokens: int = 2048,
        max_inputs_lenght: int = 10000,
        **kwargs
    ):
        if api_base is None:
            model_name = model
            if model in cls.model_aliases:
                model_name = cls.model_aliases[model]
            api_base = f"https://api-inference.huggingface.co/models/{model_name}/v1"
        start = calculate_lenght(messages)
        if start > max_inputs_lenght:
            if len(messages) > 6:
                messages = messages[:3] + messages[-3:]
            if calculate_lenght(messages) > max_inputs_lenght:
                if len(messages) > 2:
                    messages = [m for m in messages if m["role"] == "system"] + messages[-1:]
                if len(messages) > 1 and calculate_lenght(messages) > max_inputs_lenght:
                    messages = [messages[-1]]
            debug.log(f"Messages trimmed from: {start} to: {calculate_lenght(messages)}")
        async for chunk in super().create_async_generator(model, messages, api_base=api_base, max_tokens=max_tokens, **kwargs):
            yield chunk

def calculate_lenght(messages: Messages) -> int:
    return sum([len(message["content"]) + 16 for message in messages])