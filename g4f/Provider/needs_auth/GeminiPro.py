from __future__ import annotations

from ..template import OpenaiTemplate

class GeminiPro(OpenaiTemplate):
    label = "Google Gemini API"
    url = "https://ai.google.dev"
    login_url = "https://aistudio.google.com/u/0/apikey"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    backup_url = "https://g4f.space/api/gemini-v1beta"
    active_by_default = True
    working = True
    default_model = "models/gemini-2.5-flash"
    default_vision_model = default_model
    model_aliases = {
        "gemini-2.5-pro": "models/gemini-2.5-pro",
        "gemini-2.5-flash": "models/gemini-2.5-flash",
        "gemini-2.0-flash": "models/gemini-2.0-flash",
        "gemini-2.0-flash-thinking": "models/gemini-2.0-flash-thinking",
    }