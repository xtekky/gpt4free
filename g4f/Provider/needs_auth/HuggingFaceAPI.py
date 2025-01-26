from __future__ import annotations

from .OpenaiTemplate import OpenaiTemplate
from .HuggingChat import HuggingChat
from ...providers.types import Messages

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
    model_aliases = HuggingChat.model_aliases

    @classmethod
    def get_models(cls, **kwargs):
        HuggingChat.get_models()
        cls.models = HuggingChat.text_models
        cls.vision_models = HuggingChat.vision_models
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_base: str = None,
        max_tokens: int = 2048,
        **kwargs
    ):
        if api_base is None:
            model_name = model
            if model in cls.model_aliases:
                model_name = cls.model_aliases[model]
            api_base = f"https://api-inference.huggingface.co/models/{model_name}/v1"
        async for chunk in super().create_async_generator(model, messages, api_base=api_base, max_tokens=max_tokens, **kwargs):
            yield chunk