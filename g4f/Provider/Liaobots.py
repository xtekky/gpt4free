import uuid, requests

from ..typing       import Any, CreateResult
from .base_provider import BaseProvider


class Liaobots(BaseProvider):
    url: str                = "https://liaobots.com"
    supports_stream         = True
    needs_auth              = True
    supports_gpt_35_turbo   = True
    supports_gpt_4          = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:
        
        headers = {
            "authority"     : "liaobots.com",
            "content-type"  : "application/json",
            "origin"        : "https://liaobots.com",
            "referer"       : "https://liaobots.com/",
            "user-agent"    : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
            "x-auth-code"   : str(kwargs.get("auth")),
        }
        
        models = {
            "gpt-4": {
                "id": "gpt-4",
                "name": "GPT-4",
                "maxLength": 24000,
                "tokenLimit": 8000,
            },
            "gpt-3.5-turbo": {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5",
                "maxLength": 12000,
                "tokenLimit": 4000,
            },
        }
        json_data = {
            "conversationId": str(uuid.uuid4()),
            "model"         : models[model],
            "messages"      : messages,
            "key"           : "",
            "prompt"        : "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown.",
        }

        response = requests.post("https://liaobots.com/api/chat",
            headers=headers, json=json_data, stream=True)
        
        response.raise_for_status()
        for token in response.iter_content(chunk_size=2046):
            yield token.decode("utf-8")

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("auth", "str"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
