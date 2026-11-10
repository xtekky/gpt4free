from __future__ import annotations

from ..template import OpenaiTemplate


class UberLLM(OpenaiTemplate):
    label = "UberLLM"
    url = "https://uberllm.dev"
    login_url = "https://uberllm.dev/signup"
    base_url = "https://api.uberllm.dev/v1"
    working = True
    needs_auth = True
    default_model = "deepseek/deepseek-v4.1-flash"
