from __future__ import annotations

from ..template import OpenaiTemplate


class Nvidia(OpenaiTemplate):
    label = "Nvidia"
    base_url = "https://integrate.api.nvidia.com/v1"
    backup_url = "https://g4f.space/api/nvidia"
    login_url = "https://google.com"
    url = "https://build.nvidia.com"
    working = True
    active_by_default = True
    default_model = "nvidia/nemotron-3.5-lightning-30b-a3b"
    add_user = False
