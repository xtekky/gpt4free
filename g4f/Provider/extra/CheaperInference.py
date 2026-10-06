from __future__ import annotations

from ..template import OpenaiTemplate


class CheaperInference(OpenaiTemplate):
    label = "Cheaper Inference"
    url = "https://cheaperinference.com"
    login_url = "https://cheaperinference.com/signup"
    base_url = "https://api.cheaperinference.com/v1"
    working = True
    needs_auth = True
    default_model = "gpt-5.4-mini"
