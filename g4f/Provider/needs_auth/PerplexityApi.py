from __future__ import annotations

from ..template import OpenaiTemplate

class PerplexityApi(OpenaiTemplate):
    label = "Perplexity API"
    url = "https://www.perplexity.ai"
    login_url = "https://www.perplexity.ai/settings/api"
    working = True
    needs_auth = True
    api_base = "https://api.perplexity.ai"
    default_model = "llama-3-sonar-large-32k-online"
    models = [
        "llama-3-sonar-small-32k-chat",
        default_model,
        "llama-3-sonar-large-32k-chat",
        "llama-3-sonar-large-32k-online",
        "llama-3-8b-instruct",
        "llama-3-70b-instruct",
    ]