from __future__ import annotations

from ..template import OpenaiTemplate
from ...config import DEFAULT_MODEL

class Groq(OpenaiTemplate):
    url = "https://console.groq.com/playground"
    login_url = "https://console.groq.com/keys"
    base_url = "https://api.groq.com/openai/v1"
    backup_url = "https://g4f.space/api/groq"
    working = True
    active_by_default = True
    default_model = DEFAULT_MODEL
    model_aliases = {
        "mixtral-8x7b": "mixtral-8x7b-32768",
        "llama2-70b": "llama2-70b-4096",
        "moonshotai/Kimi-K2-Instruct": "moonshotai/kimi-k2-Instruct"
    }