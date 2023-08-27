import urllib.parse, json

from curl_cffi      import requests
from ..typing       import Any, CreateResult
from .base_provider import BaseProvider


class You(BaseProvider):
    url                   = "https://you.com"
    working               = True
    supports_gpt_35_turbo = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:
        
        url_param = _create_url_param(messages, kwargs.get("history", []))
        headers   = _create_header()
        
        response = requests.get(f"https://you.com/api/streamingSearch?{url_param}",
            headers=headers, impersonate="chrome107")
        
        response.raise_for_status()
        
        start = 'data: {"youChatToken": '
        for line in response.content.splitlines():
            line = line.decode('utf-8')
            if line.startswith(start):
                yield json.loads(line[len(start): -1])

def _create_url_param(messages: list[dict[str, str]], history: list[dict[str, str]]):
    prompt = ""
    for message in messages:
        prompt += "%s: %s\n" % (message["role"], message["content"])
    prompt += "assistant:"
    chat = _convert_chat(history)
    param = {"q": prompt, "domain": "youchat", "chat": chat}
    return urllib.parse.urlencode(param)


def _convert_chat(messages: list[dict[str, str]]):
    message_iter = iter(messages)
    return [
        {"question": user["content"], "answer": assistant["content"]}
        for user, assistant in zip(message_iter, message_iter)
    ]


def _create_header():
    return {
        "accept": "text/event-stream",
        "referer": "https://you.com/search?fromSearchBar=true&tbm=youchat",
    }