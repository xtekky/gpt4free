from __future__ import annotations

from ..template import OpenaiTemplate


class Nvidia(OpenaiTemplate):
    base_url = "https://integrate.api.nvidia.com/v1"
    backup_url = "https://nvidia.g4f.dev/v1"
    login_url = "https://google.com"
    url = "https://build.nvidia.com"
    working = True
    active_by_default = True
    add_user = False
