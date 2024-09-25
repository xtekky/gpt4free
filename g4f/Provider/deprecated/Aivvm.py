from __future__ import annotations

import requests
import json

from ..base_provider import AbstractProvider
from ...typing import CreateResult, Messages

# to recreate this easily, send a post request to https://chat.aivvm.com/api/models
models = {
    'gpt-3.5-turbo': {'id': 'gpt-3.5-turbo', 'name': 'GPT-3.5'},
    'gpt-3.5-turbo-0613': {'id': 'gpt-3.5-turbo-0613', 'name': 'GPT-3.5-0613'},
    'gpt-3.5-turbo-16k': {'id': 'gpt-3.5-turbo-16k', 'name': 'GPT-3.5-16K'},
    'gpt-3.5-turbo-16k-0613': {'id': 'gpt-3.5-turbo-16k-0613', 'name': 'GPT-3.5-16K-0613'},
    'gpt-4': {'id': 'gpt-4', 'name': 'GPT-4'},
    'gpt-4-0613': {'id': 'gpt-4-0613', 'name': 'GPT-4-0613'},
    'gpt-4-32k': {'id': 'gpt-4-32k', 'name': 'GPT-4-32K'},
    'gpt-4-32k-0613': {'id': 'gpt-4-32k-0613', 'name': 'GPT-4-32K-0613'},
}

class Aivvm(AbstractProvider):
    url                   = 'https://chat.aivvm.com'
    supports_stream       = True
    working               = False
    supports_gpt_35_turbo = True
    supports_gpt_4        = True

    @classmethod
    def create_completion(cls,
        model: str,
        messages: Messages,
        stream: bool,
        **kwargs
    ) -> CreateResult:
        if not model:
            model = "gpt-3.5-turbo"
        elif model not in models:
            raise ValueError(f"Model is not supported: {model}")

        json_data = {
            "model"       : models[model],
            "messages"    : messages,
            "key"         : "",
            "prompt"      : kwargs.get("system_message", "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown."),
            "temperature" : kwargs.get("temperature", 0.7)
        }

        data = json.dumps(json_data)

        headers = {
            "accept"            : "text/event-stream",
            "accept-language"   : "en-US,en;q=0.9",
            "content-type"      : "application/json",
            "content-length"    : str(len(data)),
            "sec-ch-ua"         : "\"Chrome\";v=\"117\", \"Not;A=Brand\";v=\"8\", \"Chromium\";v=\"117\"",
            "sec-ch-ua-mobile"  : "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest"    : "empty",
            "sec-fetch-mode"    : "cors",
            "sec-fetch-site"    : "same-origin",
            "sec-gpc"           : "1",
            "referrer"          : "https://chat.aivvm.com/",
            "user-agent"        : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        }

        response = requests.post("https://chat.aivvm.com/api/chat", headers=headers, data=data, stream=True)
        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=4096):
            try:
                yield chunk.decode("utf-8")
            except UnicodeDecodeError:
                yield chunk.decode("unicode-escape")