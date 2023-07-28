import re
import urllib.parse

from curl_cffi import requests

from ..typing import Any, CreateResult
from .base_provider import BaseProvider


class You(BaseProvider):
    url = "https://you.com"
    working = True
    supports_gpt_35_turbo = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs: Any,
    ) -> CreateResult:
        url_param = _create_url_param(messages)
        headers = _create_header()
        url = f"https://you.com/api/streamingSearch?{url_param}"
        response = requests.get(
            url,
            headers=headers,
            impersonate="chrome107",
        )
        response.raise_for_status()
        yield _parse_output(response.text)


def _create_url_param(messages: list[dict[str, str]]):
    prompt = messages.pop()["content"]
    chat = _convert_chat(messages)
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


def _parse_output(output: str) -> str:
    regex = r"^data:\s{\"youChatToken\": \"(.*)\"}$"
    tokens = [token for token in re.findall(regex, output, re.MULTILINE)]
    return "".join(tokens)
