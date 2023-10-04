from __future__ import annotations

import uuid

import requests

from ...typing import Any, CreateResult
from ..base_provider import BaseProvider


class V50(BaseProvider):
    url                     = 'https://p5.v50.ltd'
    supports_gpt_35_turbo   = True
    supports_stream         = False
    needs_auth              = False
    working                 = False

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:
        
        conversation = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        conversation += "\nassistant: "

        payload = {
            "prompt"        : conversation,
            "options"       : {},
            "systemMessage" : ".",
            "temperature"   : kwargs.get("temperature", 0.4),
            "top_p"         : kwargs.get("top_p", 0.4),
            "model"         : model,
            "user"          : str(uuid.uuid4())
        }
        
        headers = {
            'authority'         : 'p5.v50.ltd',
            'accept'            : 'application/json, text/plain, */*',
            'accept-language'   : 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
            'content-type'      : 'application/json',
            'origin'            : 'https://p5.v50.ltd',
            'referer'           : 'https://p5.v50.ltd/',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest'    : 'empty',
            'sec-fetch-mode'    : 'cors',
            'sec-fetch-site'    : 'same-origin',
            'user-agent'        : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
        }
        response = requests.post("https://p5.v50.ltd/api/chat-process", 
                                json=payload, headers=headers, proxies=kwargs['proxy'] if 'proxy' in kwargs else {})
        
        if "https://fk1.v50.ltd" not in response.text:
            yield response.text

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