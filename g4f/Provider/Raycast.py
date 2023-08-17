import json

import requests

from ..Provider.base_provider import BaseProvider
from ..typing import Any, CreateResult


class Raycast(BaseProvider):
    url = "https://backend.raycast.com/api/v1/ai/chat_completions"
    working = True
    supports_stream = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    needs_auth = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs: Any,
    ) -> CreateResult:
        auth = kwargs.get("auth")

        headers = {
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Authorization": f"Bearer {auth}",
            "Content-Type": "application/json",
            "User-Agent": "Raycast/0 CFNetwork/1410.0.3 Darwin/22.6.0",
        }
        parsed_messages: list[dict[str, Any]] = []
        for message in messages:
            parsed_messages.append(
                {"author": message["role"], "content": {"text": message["content"]}}
            )
        data = {
            "debug": False,
            "locale": "en-CN",
            "messages": parsed_messages,
            "model": model,
            "provider": "openai",
            "source": "ai_chat",
            "system_instruction": "markdown",
            "temperature": 0.5,
        }
        url = "https://backend.raycast.com/api/v1/ai/chat_completions"
        response = requests.post(url, headers=headers, json=data, stream=True)
        for token in response.iter_lines():
            if b"data: " not in token:
                continue
            completion_chunk = json.loads(token.decode().replace("data: ", ""))
            token = completion_chunk["text"]
            if token != None:
                yield token
