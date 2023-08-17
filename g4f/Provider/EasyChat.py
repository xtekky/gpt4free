import json

import requests

from ..typing import Any, CreateResult
from .base_provider import BaseProvider


class EasyChat(BaseProvider):
    url = "https://free.easychat.work"
    supports_stream = True
    supports_gpt_35_turbo = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs: Any,
    ) -> CreateResult:
        active_servers = [
            "https://chat10.fastgpt.me",
            "https://chat9.fastgpt.me",
            "https://chat1.fastgpt.me",
            "https://chat2.fastgpt.me",
            "https://chat3.fastgpt.me",
            "https://chat4.fastgpt.me",
        ]
        server = active_servers[kwargs.get("active_server", 0)]
        headers = {
            "authority": f"{server}".replace("https://", ""),
            "accept": "text/event-stream",
            "accept-language": "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3,fa=0.2",
            "content-type": "application/json",
            "origin": f"{server}",
            "referer": f"{server}/",
            "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "x-requested-with": "XMLHttpRequest",
        }

        json_data = {
            "messages": messages,
            "stream": stream,
            "model": model,
            "temperature": kwargs.get("temperature", 0.5),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "top_p": kwargs.get("top_p", 1),
        }

        session = requests.Session()
        # init cookies from server
        session.get(f"{server}/")

        response = session.post(
            f"{server}/api/openai/v1/chat/completions",
            headers=headers,
            json=json_data,
        )

        response.raise_for_status()
        print(response.text)
        for chunk in response.iter_lines():
            if b"content" in chunk:
                data = json.loads(chunk.decode().split("data: ")[1])
                yield data["choices"][0]["delta"]["content"]

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
            ("active_server", "int"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
