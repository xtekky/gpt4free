from __future__ import annotations

from .template import OpenaiTemplate
from ..config import DEFAULT_MODEL

class ApiAirforce(OpenaiTemplate):
    label = "Api.Airforce"
    url = "https://api.airforce"
    login_url = "https://panel.api.airforce/dashboard"
    api_base = "https://api.airforce/v1"
    working = True
    active_by_default = True
    use_image_size = True