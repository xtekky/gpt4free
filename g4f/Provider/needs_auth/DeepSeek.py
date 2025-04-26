from __future__ import annotations

from .OpenaiAPI import OpenaiAPI

class DeepSeek(OpenaiAPI):
    label = "DeepSeek"
    url = "https://platform.deepseek.com"
    login_url = "https://platform.deepseek.com/api_keys"
    working = True
    api_base = "https://api.deepseek.com"
    needs_auth = True
    supports_stream = True
    supports_message_history = True
    default_model = "deepseek-chat"
    fallback_models = [default_model]
