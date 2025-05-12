from __future__ import annotations

from ..template import OpenaiTemplate

class Glider(OpenaiTemplate):
    label = "Glider"
    url = "https://glider.so"
    api_endpoint = "https://glider.so/api/chat"
    working = False

    default_model = 'chat-llama-3-1-70b'
    models = [
        'chat-llama-3-1-70b',
        'chat-llama-3-1-8b',
        'chat-llama-3-2-3b',
        'deepseek-ai/DeepSeek-R1'
    ]

    model_aliases = {
        "llama-3.1-70b": "chat-llama-3-1-70b",
        "llama-3.1-8b": "chat-llama-3-1-8b",
        "llama-3.2-3b": "chat-llama-3-2-3b",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    }
