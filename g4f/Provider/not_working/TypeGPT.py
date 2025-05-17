from __future__ import annotations

import requests

from ..template import OpenaiTemplate
from ...errors import ModelNotFoundError
from ... import debug

class TypeGPT(OpenaiTemplate):
    label = "TypeGpt"
    url = "https://chat.typegpt.net"
    api_base = "https://chat.typegpt.net/api/openai/v1"
    working = False
    headers = {
        "accept": "application/json, text/event-stream",
        "accept-language": "de,en-US;q=0.9,en;q=0.8",
        "content-type": "application/json",
        "priority": "u=1, i",
        "sec-ch-ua": "\"Not(A:Brand\";v=\"99\", \"Google Chrome\";v=\"133\", \"Chromium\";v=\"133\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Linux\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "referer": "https://chat.typegpt.net/",
    }
    
    default_model = 'gpt-4o-mini-2024-07-18'
    default_vision_model = default_model
    vision_models = ['gpt-3.5-turbo', 'gpt-3.5-turbo-202201', default_vision_model, "o3-mini"]
    fallback_models = vision_models + ["deepseek-r1", "deepseek-v3", "evil"]
    image_models = ["Image-Generator"]
    model_aliases = {
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "evil": "uncensored-r1",
    }

    @classmethod
    def get_models(cls, **kwargs):
        if not cls.models:
            try:
                cls.models = requests.get(f"{cls.url}/api/config").json()["customModels"].split(",")
                cls.models = [model.split("@")[0].strip("+") for model in cls.models if not model.startswith("-") and model not in cls.image_models]
            except Exception as e:
                cls.models = cls.fallback_models
                debug.log(f"Error fetching models: {e}")
        return cls.models
