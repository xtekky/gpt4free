import json

from curl_cffi import requests

from ..typing import Any, CreateResult
from .base_provider import BaseProvider


class Theb(BaseProvider):
    url = "https://theb.ai"
    working = False
    supports_stream = True
    supports_gpt_35_turbo = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs: Any,
    ) -> CreateResult:
        prompt = messages[-1]["content"]

        headers = {
            "accept": "application/json, text/plain, */*",
            "content-type": "application/json",
        }

        json_data: dict[str, Any] = {"prompt": prompt, "options": {}}
        response = requests.post(
            "https://chatbot.theb.ai/api/chat-process",
            headers=headers,
            json=json_data,
            impersonate="chrome110",
        )
        response.raise_for_status()
        line = response.text.splitlines()[-1]
        text = json.loads(line)["text"]
        yield text
