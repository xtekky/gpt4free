from __future__ import annotations

import json

import requests

from ...typing import Any, CreateResult
from ..base_provider import AbstractProvider


class Lockchat(AbstractProvider):
    url: str              = "http://supertest.lockchat.app"
    supports_stream       = True
    supports_gpt_35_turbo = True
    supports_gpt_4        = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:

        temperature = float(kwargs.get("temperature", 0.7))
        payload = {
            "temperature": temperature,
            "messages"   : messages,
            "model"      : model,
            "stream"     : True,
        }

        headers = {
            "user-agent": "ChatX/39 CFNetwork/1408.0.4 Darwin/22.5.0",
        }
        response = requests.post("http://supertest.lockchat.app/v1/chat/completions",
                                 json=payload, headers=headers, stream=True)

        response.raise_for_status()
        for token in response.iter_lines():
            if b"The model: `gpt-4` does not exist" in token:
                print("error, retrying...")
                
                Lockchat.create_completion(
                    model       = model,
                    messages    = messages,
                    stream      = stream,
                    temperature = temperature,
                    **kwargs)

            if b"content" in token:
                token = json.loads(token.decode("utf-8").split("data: ")[1])
                token = token["choices"][0]["delta"].get("content")

                if token:
                    yield (token)