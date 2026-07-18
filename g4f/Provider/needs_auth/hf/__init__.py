from __future__ import annotations

from ...template.OpenaiTemplate import OpenaiTemplate
from .HuggingFaceMedia import HuggingFaceMedia
from .HuggingChat import HuggingChat

class HuggingFace(OpenaiTemplate):
    url = "https://huggingface.co"
    base_url = "https://router.huggingface.co/v1"
    login_url = "https://huggingface.co/settings/tokens"
    working = True
    active_by_default = True
    quota_url = "https://huggingface.co/api/whoami-v2"
