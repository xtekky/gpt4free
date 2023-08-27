import hashlib
import json
import time
import uuid
from datetime import datetime

import requests

from ..typing import SHA256, Any, CreateResult
from .base_provider import BaseProvider

class Ails(BaseProvider):
    url: str              = "https://ai.ls"
    working               = True
    supports_stream       = True
    supports_gpt_35_turbo = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:
        
        headers = {
            "authority": "api.caipacity.com",
            "accept": "*/*",
            "accept-language": "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "authorization": "Bearer free",
            "client-id": str(uuid.uuid4()),
            "client-v": _get_client_v(),
            "content-type": "application/json",
            "origin": "https://ai.ls",
            "referer": "https://ai.ls/",
            "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "cross-site",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        }

        timestamp = _format_timestamp(int(time.time() * 1000))
        sig = {
            "d": datetime.now().strftime("%Y-%m-%d"),
            "t": timestamp,
            "s": _hash({"t": timestamp, "m": messages[-1]["content"]}),
        }

        json_data = json.dumps(
            separators=(",", ":"),
            obj={
                "model": "gpt-3.5-turbo",
                "temperature": kwargs.get("temperature", 0.6),
                "stream": True,
                "messages": messages,
            }
            | sig,
        )

        response = requests.post(
            "https://api.caipacity.com/v1/chat/completions",
            headers=headers,
            data=json_data,
            stream=True,
        )
        response.raise_for_status()

        for token in response.iter_lines():
            if b"content" in token:
                completion_chunk = json.loads(token.decode().replace("data: ", ""))
                token = completion_chunk["choices"][0]["delta"].get("content")
                if "ai.ls" in token.lower() or "ai.ci" in token.lower():
                    raise Exception("Response Error: " + token)
                if token != None:
                    yield token

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("temperature", "float"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"


def _hash(json_data: dict[str, str]) -> SHA256:
    base_string: str = "%s:%s:%s:%s" % (
        json_data["t"],
        json_data["m"],
        "WI,2rU#_r:r~aF4aJ36[.Z(/8Rv93Rf",
        len(json_data["m"]),
    )

    return SHA256(hashlib.sha256(base_string.encode()).hexdigest())


def _format_timestamp(timestamp: int) -> str:
    e = timestamp
    n = e % 10
    r = n + 1 if n % 2 == 0 else n
    return str(e - n + r)


def _get_client_v():
    response = requests.get("https://ai.ls/?chat=1")
    response.raise_for_status()
    js_path = response.text.split('crossorigin href="')[1].split('"')[0]

    response = requests.get("https://ai.ls" + js_path)
    response.raise_for_status()
    return response.text.split('G4="')[1].split('"')[0]
