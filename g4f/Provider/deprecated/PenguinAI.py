from __future__ import annotations
import requests

from ..template import OpenaiTemplate
from ...requests import raise_for_status
from ... import debug

class PenguinAI(OpenaiTemplate):
    label = "PenguinAI"
    url = "https://penguinai.tech"
    api_base = "https://api.penguinai.tech/v1"
    working = False
    active_by_default = False

    default_model = "gpt-3.5-turbo"
    default_vision_model = "gpt-4o"

    # in reality, it uses pollinations
    image_models = ["flux"]

    @classmethod
    def get_models(cls, api_key: str = None, api_base: str = None) -> list[str]:
        if not cls.models:
            try:
                headers = {}
                if api_base is None:
                    api_base = cls.api_base
                if api_key is None and cls.api_key is not None:
                    api_key = cls.api_key
                if api_key is not None:
                    headers["authorization"] = f"Bearer {api_key}"
                response = requests.get(f"{api_base}/models", headers=headers, verify=cls.ssl)
                raise_for_status(response)
                data = response.json()
                data = data.get("data") if isinstance(data, dict) else data
                cls.image_models = [model.get("id") for model in data if "image" in model.get("type")]
                cls.models = [model.get("id") for model in data]
                if cls.sort_models:
                    cls.models.sort()
                    
                cls.vision_models = []
                vision_model_prefixes = ["vision", "multimodal", "o1", "o3", "o4", "gpt-4", "claude-3", "claude-opus", "claude-sonnet"]
                for model in cls.models:
                    for tag in vision_model_prefixes:
                        if tag in model and not "search" in model:
                            cls.vision_models.append(model)
            except Exception as e:
                debug.error(e)
                return cls.fallback_models
        return cls.models