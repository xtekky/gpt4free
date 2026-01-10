from __future__ import annotations

from ..template import OpenaiTemplate

class GeminiPro(OpenaiTemplate):
    label = "Google Gemini API"
    url = "https://ai.google.dev"
    login_url = "https://aistudio.google.com/u/0/apikey"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    backup_url = "https://g4f.dev/custom/srv_mjnryskw9fe0567fa267"
    active_by_default = True
    working = True
    default_model = "gemini-2.5-flash"
    default_vision_model = default_model
    fallback_models = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemma-3-1b-it",
        "gemma-3-12b-it",
        "gemma-3-27b-it",
        "gemma-3-4b-it",
        "gemma-3n-e2b-it",
        "gemma-3n-e4b-it",
    ]