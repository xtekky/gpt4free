from __future__ import annotations

from ..template import OpenaiTemplate
from ...config import DEFAULT_MODEL

class Groq(OpenaiTemplate):
    url = "https://console.groq.com/playground"
    login_url = "https://console.groq.com/keys"
    api_base = "https://api.groq.com/openai/v1"
    working = True
    needs_auth = True
    active_by_default = True
    default_model = DEFAULT_MODEL
    fallback_models = [
        "distil-whisper-large-v3-en",
        "gemma2-9b-it",
        "gemma-7b-it",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-90b-vision-preview",
        "llama-guard-3-8b",
        "llava-v1.5-7b-4096-preview",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "whisper-large-v3",
        "whisper-large-v3-turbo",
    ]
    model_aliases = {
        "mixtral-8x7b": "mixtral-8x7b-32768",
        "llama2-70b": "llama2-70b-4096",
        "moonshotai/Kimi-K2-Instruct": "moonshotai/kimi-k2-Instruct"
    }