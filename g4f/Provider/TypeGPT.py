from __future__ import annotations

from .template import OpenaiTemplate

class TypeGPT(OpenaiTemplate):
    label = "TypeGpt"
    url = "https://chat.typegpt.net"
    api_endpoint = "https://chat.typegpt.net/api/openai/typegpt/v1/chat/completions"
    working = True
    
    default_model = "gpt-4o-mini-2024-07-18"
    models = [
       default_model, "o1", "o3-mini", "gemini-1.5-flash", "deepseek-r1", "deepseek-v3", "gemini-pro", "evil"
    ]
    model_aliases = {
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    }