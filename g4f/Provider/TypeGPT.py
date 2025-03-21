from __future__ import annotations

from .template import OpenaiTemplate

class TypeGPT(OpenaiTemplate):
    label = "TypeGpt"
    url = "https://chat.typegpt.net"
    api_base = "https://chat.typegpt.net/api/openai/typegpt/v1"
    working = True
    
    default_model = 'gpt-4o-mini-2024-07-18'
    default_vision_model = default_model
    vision_models = ['gpt-3.5-turbo', 'gpt-3.5-turbo-202201', default_vision_model, "o3-mini"]
    models = vision_models + ["deepseek-r1", "deepseek-v3", "evil", "o1"]
    model_aliases = {
        "gpt-3.5-turbo": "gpt-3.5-turbo-202201",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    }
