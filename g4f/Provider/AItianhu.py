import json

import requests

from ..typing import Any, CreateResult
from .base_provider import BaseProvider


class AItianhu(BaseProvider):
    url = "https://www.aitianhu.com/"
    working = False
    supports_gpt_35_turbo = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:
        
        base = ""
        for message in messages:
            base += "%s: %s\n" % (message["role"], message["content"])
        base += "assistant:"

        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        data: dict[str, Any] = {
            "prompt": base,
            "options": {},
            "systemMessage": "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown.",
            "temperature": kwargs.get("temperature", 0.8),
            "top_p": kwargs.get("top_p", 1),
        }
        url = "https://www.aitianhu.com/api/chat-process"
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        lines = response.text.strip().split("\n")
        res = json.loads(lines[-1])
        yield res["text"]

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("temperature", "float"),
            ("top_p", "int"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
