from __future__ import annotations

from abc import abstractmethod

class ResponseType:
    @abstractmethod
    def __str__(self) -> str:
        pass

class FinishReason():
    def __init__(self, reason: str):
        self.reason = reason

    def __str__(self) -> str:
        return ""

class Sources(ResponseType):
    def __init__(self, sources: list[dict[str, str]]) -> None:
        self.list = sources

    def __str__(self) -> str:
        return "\n\n" + ("\n".join([f"{idx+1}. [{link['title']}]({link['url']})" for idx, link in enumerate(self.list)]))

class BaseConversation(ResponseType):
    def __str__(self) -> str:
        return ""

class SynthesizeData(ResponseType):
    def __init__(self, provider: str, data: dict):
        self.provider = provider
        self.data = data

    def to_json(self) -> dict:
        return {
            **self.__dict__
        }

    def __str__(self) -> str:
        return ""