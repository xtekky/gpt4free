from __future__ import annotations

from ..template import OpenaiTemplate
from ...config import DEFAULT_MODEL
import os

class Nvidia(OpenaiTemplate):
    label = "Nvidia"
    api_base = "https://integrate.api.nvidia.com/v1"
    login_url = "https://google.com"
    url = "https://build.nvidia.com"
    working = True
    needs_auth = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    default_model = DEFAULT_MODEL.split("/")[-1]

    @classmethod
    def get_model(cls, model: str, **kwargs) -> str:
        return super().get_model(model, **kwargs)   