from __future__ import annotations

from .OpenaiAPI import OpenaiAPI
from ..HuggingChat import HuggingChat
from ...typing import AsyncResult, Messages

class HuggingFace2(OpenaiAPI):
    label = "HuggingFace (Inference API)"
    url = "https://huggingface.co"
    working = True
    default_model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    default_vision_model = default_model
    models = [
        *HuggingChat.models
    ]

    @classmethod
    def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_base: str = "https://api-inference.huggingface.co/v1",
        max_tokens: int = 500,
        **kwargs
    ) -> AsyncResult:
        return super().create_async_generator(
            model, messages, api_base=api_base, max_tokens=max_tokens, **kwargs
        )