import requests

from ..typing       import Any, CreateResult
from .base_provider import BaseProvider


class Opchatgpts(BaseProvider):
    url                   = "https://opchatgpts.net"
    working               = True
    supports_gpt_35_turbo = True

    @staticmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:
        
        temperature   = kwargs.get("temperature", 0.8)
        max_tokens    = kwargs.get("max_tokens", 1024)
        system_prompt = kwargs.get(
            "system_prompt",
            "Converse as if you were an AI assistant. Be friendly, creative.")
        
        payload = _create_payload(
            messages        = messages,
            temperature     = temperature,
            max_tokens      = max_tokens,
            system_prompt   = system_prompt)

        response = requests.post("https://opchatgpts.net/wp-json/ai-chatbot/v1/chat", json=payload)
        
        response.raise_for_status()
        yield response.json()["reply"]


def _create_payload(
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int, system_prompt: str) -> dict:
    
    return {
        "env"             : "chatbot",
        "session"         : "N/A",
        "prompt"          : "\n",
        "context"         : system_prompt,
        "messages"        : messages,
        "newMessage"      : messages[::-1][0]["content"],
        "userName"        : '<div class="mwai-name-text">User:</div>',
        "aiName"          : '<div class="mwai-name-text">AI:</div>',
        "model"           : "gpt-3.5-turbo",
        "temperature"     : temperature,
        "maxTokens"       : max_tokens,
        "maxResults"      : 1,
        "apiKey"          : "",
        "service"         : "openai",
        "embeddingsIndex" : "",
        "stop"            : "",
    }
