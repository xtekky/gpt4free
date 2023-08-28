import base64, os, re, requests

from ..typing       import Any, CreateResult
from .base_provider import BaseProvider


class ChatgptLogin(BaseProvider):
    url                   = "https://opchatgpts.net"
    supports_gpt_35_turbo = True
    working               = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:
        
        headers = {
            "authority"          : "chatgptlogin.ac",
            "accept"             : "*/*",
            "accept-language"    : "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "content-type"       : "application/json",
            "origin"             : "https://opchatgpts.net",
            "referer"            : "https://opchatgpts.net/chatgpt-free-use/",
            "sec-ch-ua"          : '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
            "sec-ch-ua-mobile"   : "?0",
            "sec-ch-ua-platform" : '"Windows"',
            "sec-fetch-dest"     : "empty",
            "sec-fetch-mode"     : "cors",
            "sec-fetch-site"     : "same-origin",
            "user-agent"         : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            "x-wp-nonce"         : _get_nonce(),
        }

        conversation = _transform(messages)

        json_data = {
            "env"            : "chatbot",
            "session"        : "N/A",
            "prompt"         : "Converse as if you were an AI assistant. Be friendly, creative.",
            "context"        : "Converse as if you were an AI assistant. Be friendly, creative.",
            "messages"       : conversation,
            "newMessage"     : messages[-1]["content"],
            "userName"       : '<div class="mwai-name-text">User:</div>',
            "aiName"         : '<div class="mwai-name-text">AI:</div>',
            "model"          : "gpt-3.5-turbo",
            "temperature"    : kwargs.get("temperature", 0.8),
            "maxTokens"      : 1024,
            "maxResults"     : 1,
            "apiKey"         : "",
            "service"        : "openai",
            "embeddingsIndex": "",
            "stop"           : "",
            "clientId"       : os.urandom(6).hex()
        }

        response = requests.post("https://opchatgpts.net/wp-json/ai-chatbot/v1/chat",
            headers=headers, json=json_data)
        
        response.raise_for_status()
        yield response.json()["reply"]

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


def _get_nonce() -> str:
    res = requests.get("https://opchatgpts.net/chatgpt-free-use/",
        headers = {
            "Referer"   : "https://opchatgpts.net/chatgpt-free-use/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"})

    result = re.search(
        r'class="mwai-chat mwai-chatgpt">.*<span>Send</span></button></div></div></div> <script defer src="(.*?)">',
        res.text)
    
    if result is None:
        return ""
    
    src            = result.group(1)
    decoded_string = base64.b64decode(src.split(",")[-1]).decode("utf-8")
    result         = re.search(r"let restNonce = '(.*?)';", decoded_string)

    return "" if result is None else result.group(1)


def _transform(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "id"     : os.urandom(6).hex(),
            "role"   : message["role"],
            "content": message["content"],
            "who"    : "AI: " if message["role"] == "assistant" else "User: ",
            "html"   : _html_encode(message["content"]),
        }
        for message in messages
    ]


def _html_encode(string: str) -> str:
    table = {
        '"' : "&quot;",
        "'" : "&#39;",
        "&" : "&amp;",
        ">" : "&gt;",
        "<" : "&lt;",
        "\n": "<br>",
        "\t": "&nbsp;&nbsp;&nbsp;&nbsp;",
        " " : "&nbsp;",
    }

    for key in table:
        string = string.replace(key, table[key])

    return string
