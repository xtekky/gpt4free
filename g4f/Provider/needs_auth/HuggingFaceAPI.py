from __future__ import annotations

from .OpenaiAPI import OpenaiAPI
from .HuggingChat import HuggingChat
from ...providers.types import Messages

class HuggingFaceAPI(OpenaiAPI):
    label = "HuggingFace (Inference API)"
    parent = "HuggingFace"
    url = "https://api-inference.huggingface.com"
    api_base = "https://api-inference.huggingface.co/v1"
    working = True

    default_model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    default_vision_model = default_model

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
        **kwargs
    ):
        if api_base is None:
            api_base = f"https://api-inference.huggingface.co/models/{model}/v1"
        async for chunk in super().create_async_generator(model, messages, api_base=api_base, **kwargs):
            yield chunk