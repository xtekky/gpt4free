import requests

from ..typing import Any, CreateResult
from .base_provider import BaseProvider


class Aichat(BaseProvider):
    url                   = "https://chat-gpt.org/chat"
    working               = True
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
            "authority": "chat-gpt.org",
            "accept": "*/*",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": "https://chat-gpt.org",
            "pragma": "no-cache",
            "referer": "https://chat-gpt.org/chat",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
        }

        json_data = {
            "message": base,
            "temperature": kwargs.get('temperature', 0.5),
            "presence_penalty": 0,
            "top_p": kwargs.get('top_p', 1),
            "frequency_penalty": 0,
        }

        response = requests.post(
            "https://chat-gpt.org/api/text",
            headers=headers,
            json=json_data,
        )
        response.raise_for_status()
        if not response.json()['response']:
            raise Exception("Error Response: " + response.json())
        yield response.json()["message"]
