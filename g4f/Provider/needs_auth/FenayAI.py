from __future__ import annotations

from ..template import OpenaiTemplate
from ...config import DEFAULT_MODEL

class FenayAI(OpenaiTemplate):
    url = "https://fenayai.com"
    login_url = "https://fenayai.com/dashboard"
    api_base = "https://fenayai.com/v1"
    working = True
    needs_auth = True
    default_model = DEFAULT_MODEL.split("/")[-1]

    @classmethod
    def get_model(cls, model: str, **kwargs) -> str:
        return super().get_model(model.split("/")[-1], **kwargs)   