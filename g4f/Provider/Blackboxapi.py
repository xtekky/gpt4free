from __future__ import annotations

from .template import OpenaiTemplate

class Blackboxapi(OpenaiTemplate):
    label = "BlackBox API"
    url = "https://www.blackboxapi.com"
    api_base = "https://www.blackboxapi.com"
    api_endpoint = "https://www.blackboxapi.com/chat/completions"
    api_key = "API_KEY"
    
    working = True
    needs_auth = False
    
    default_model = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
    models = [default_model]
    
    model_aliases = {
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    }
