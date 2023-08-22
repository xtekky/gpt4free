import time

import requests

from ..typing import Any, CreateResult
from .base_provider import BaseProvider


class Acytoo(BaseProvider):
    url = "https://chat.acytoo.com/api/completions"
    working = True
    supports_gpt_35_turbo = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs: Any,
    ) -> CreateResult:
        headers = _create_header()
        payload = _create_payload(messages, kwargs.get('temperature', 0.5))

        url = "https://chat.acytoo.com/api/completions"
        response = requests.post(url=url, headers=headers, json=payload)
        response.raise_for_status()
        response.encoding = "utf-8"
        yield response.text


def _create_header():
    return {
        "accept": "*/*",
        "content-type": "application/json",
    }


def _create_payload(messages: list[dict[str, str]], temperature):
    payload_messages = [
        message | {"createdAt": int(time.time()) * 1000} for message in messages
    ]
    return {
        "key": "",
        "model": "gpt-3.5-turbo",
        "messages": payload_messages,
        "temperature": temperature,
        "password": "",
    }
