from __future__ import annotations

from .OpenaiAPI import OpenaiAPI

class PerplexityApi(OpenaiAPI):
    label = "Perplexity API"
    url = "https://www.perplexity.ai"
    working = True
    api_base = "https://api.perplexity.ai"
    default_model = "llama-3-sonar-large-32k-online"
    models = [
        "llama-3-sonar-small-32k-chat",
        "llama-3-sonar-small-32k-online",
        "llama-3-sonar-large-32k-chat",
        "llama-3-sonar-large-32k-online",
        "llama-3-8b-instruct",
        "llama-3-70b-instruct",
    ]