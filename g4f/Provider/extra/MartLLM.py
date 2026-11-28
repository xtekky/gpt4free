from __future__ import annotations

from ..template import OpenaiTemplate


class MartLLM(OpenaiTemplate):
    label = "MartLLM"
    url = "https://martllm.com"
    login_url = "https://martllm.com/signup"
    base_url = "https://api.martllm.com/v1"
    working = True
    needs_auth = True
    default_model = "deepseek/deepseek-v4.1-flash"
