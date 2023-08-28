import requests

from ..typing       import Any, CreateResult
from .base_provider import BaseProvider


class Yqcloud(BaseProvider):
    url                     = "https://chat9.yqcloud.top/"
    working                 = True
    supports_gpt_35_turbo   = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:
        
        headers = _create_header()
        payload = _create_payload(messages)

        response = requests.post("https://api.aichatos.cloud/api/generateStream", 
                                 headers=headers, json=payload)
        
        response.raise_for_status()
        response.encoding = 'utf-8'
        yield response.text


def _create_header():
    return {
        "accept"        : "application/json, text/plain, */*",
        "content-type"  : "application/json",
        "origin"        : "https://chat9.yqcloud.top",
    }


def _create_payload(messages: list[dict[str, str]]):
    prompt = ""
    for message in messages:
        prompt += "%s: %s\n" % (message["role"], message["content"])
    prompt += "assistant:"
    
    return {
        "prompt"        : prompt,
        "network"       : True,
        "system"        : "",
        "withoutContext": False,
        "stream"        : False,
    }