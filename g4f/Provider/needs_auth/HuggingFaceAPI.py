from __future__ import annotations

from .OpenaiAPI import OpenaiAPI
from .HuggingChat import HuggingChat

class HuggingFaceAPI(OpenaiAPI):
    label = "HuggingFace (Inference API)"
    parent = "HuggingFace"
    url = "https://api-inference.huggingface.com"
    api_base = "https://api-inference.huggingface.co/v1"
    working = True
    default_model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    default_vision_model = default_model
    models = [
        *HuggingChat.models
    ]