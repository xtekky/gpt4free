from __future__ import annotations

import json
import re
import time

import requests

from ...typing import Any, CreateResult
from ..base_provider import BaseProvider


class DfeHub(BaseProvider):
    url                   = "https://chat.dfehub.com/"
    supports_stream       = True
    supports_gpt_35_turbo = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:
        
        headers = {
            "authority"         : "chat.dfehub.com",
            "accept"            : "*/*",
            "accept-language"   : "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "content-type"      : "application/json",
            "origin"            : "https://chat.dfehub.com",
            "referer"           : "https://chat.dfehub.com/",
            "sec-ch-ua"         : '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
            "sec-ch-ua-mobile"  : "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest"    : "empty",
            "sec-fetch-mode"    : "cors",
            "sec-fetch-site"    : "same-origin",
            "user-agent"        : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "x-requested-with"  : "XMLHttpRequest",
        }

        json_data = {
            "messages"          : messages,
            "model"             : "gpt-3.5-turbo",
            "temperature"       : kwargs.get("temperature", 0.5),
            "presence_penalty"  : kwargs.get("presence_penalty", 0),
            "frequency_penalty" : kwargs.get("frequency_penalty", 0),
            "top_p"             : kwargs.get("top_p", 1),
            "stream"            : True
        }
        
        response = requests.post("https://chat.dfehub.com/api/openai/v1/chat/completions",
            headers=headers, json=json_data, timeout=3)

        for chunk in response.iter_lines():
            if b"detail" in chunk:
                delay = re.findall(r"\d+\.\d+", chunk.decode())
                delay = float(delay[-1])
                time.sleep(delay)
                yield from DfeHub.create_completion(model, messages, stream, **kwargs)
            if b"content" in chunk:
                data = json.loads(chunk.decode().split("data: ")[1])
                yield (data["choices"][0]["delta"]["content"])

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("temperature", "float"),
            ("presence_penalty", "int"),
            ("frequency_penalty", "int"),
            ("top_p", "int"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
