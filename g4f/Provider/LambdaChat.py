from __future__ import annotations

from .hf.HuggingChat import HuggingChat

class LambdaChat(HuggingChat):
    label = "Lambda Chat"
    domain = "lambda.chat"
    origin = f"https://{domain}"
    url = origin
    working = True
    use_nodriver = False
    needs_auth = False

    default_model = "deepseek-llama3.3-70b"
    reasoning_model = "deepseek-r1"
    image_models = []
    fallback_models = [
        default_model,
        reasoning_model,
        "hermes-3-llama-3.1-405b-fp8",
        "llama3.1-nemotron-70b-instruct",
        "lfm-40b",
        "llama3.3-70b-instruct-fp8"
    ]
    models = fallback_models.copy()
    
    model_aliases = {
        "deepseek-v3": default_model,
        "hermes-3": "hermes-3-llama-3.1-405b-fp8",
        "nemotron-70b": "llama3.1-nemotron-70b-instruct",
        "llama-3.3-70b": "llama3.3-70b-instruct-fp8"
    }