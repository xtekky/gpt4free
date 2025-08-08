from __future__ import annotations

from ..template import OpenaiTemplate
from ...config import DEFAULT_MODEL

class Nvidia(OpenaiTemplate):
    label = "Nvidia"
    api_base = "https://integrate.api.nvidia.com/v1"
    login_url = "https://google.com"
    url = "https://build.nvidia.com"
    working = True
    active_by_default = True
    needs_auth = True
    models_needs_auth = True
    default_model = DEFAULT_MODEL
    add_user = False