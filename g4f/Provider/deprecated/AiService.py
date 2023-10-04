from __future__ import annotations

import requests

from ...typing import Any, CreateResult
from ..base_provider import BaseProvider


class AiService(BaseProvider):
    url = "https://aiservice.vercel.app/"
    working = False
    supports_gpt_35_turbo = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs: Any,
    ) -> CreateResult:
        base = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        base += "\nassistant: "

        headers = {
            "accept": "*/*",
            "content-type": "text/plain;charset=UTF-8",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "Referer": "https://aiservice.vercel.app/chat",
        }
        data = {"input": base}
        url = "https://aiservice.vercel.app/api/chat/answer"
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        yield response.json()["data"]
