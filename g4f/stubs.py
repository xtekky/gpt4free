
from __future__ import annotations

from typing import Union

class Model():
    ...

class ChatCompletion(Model):
    def __init__(
        self,
        content: str,
        finish_reason: str,
        completion_id: str = None,
        created: int = None
    ):
        self.id: str = f"chatcmpl-{completion_id}" if completion_id else None
        self.object: str = "chat.completion"
        self.created: int = created
        self.model: str = None
        self.provider: str = None
        self.choices = [ChatCompletionChoice(ChatCompletionMessage(content), finish_reason)]
        self.usage: dict[str, int] = {
            "prompt_tokens": 0, #prompt_tokens,
            "completion_tokens": 0, #completion_tokens,
            "total_tokens": 0, #prompt_tokens + completion_tokens,
        }

    def to_json(self):
        return {
            **self.__dict__,
            "choices": [choice.to_json() for choice in self.choices]
        }

class ChatCompletionChunk(Model):
    def __init__(
        self,
        content: str,
        finish_reason: str,
        completion_id: str = None,
        created: int = None
    ):
        self.id: str = f"chatcmpl-{completion_id}" if completion_id else None
        self.object: str = "chat.completion.chunk"
        self.created: int = created
        self.model: str = None
        self.provider: str = None
        self.choices = [ChatCompletionDeltaChoice(ChatCompletionDelta(content), finish_reason)]

    def to_json(self):
        return {
            **self.__dict__,
            "choices": [choice.to_json() for choice in self.choices]
        }

class ChatCompletionMessage(Model):
    def __init__(self, content: Union[str, None]):
        self.role = "assistant"
        self.content = content

    def to_json(self):
        return self.__dict__

class ChatCompletionChoice(Model):
    def __init__(self, message: ChatCompletionMessage, finish_reason: str):
        self.index = 0
        self.message = message
        self.finish_reason = finish_reason

    def to_json(self):
        return {
            **self.__dict__,
            "message": self.message.to_json()
        }

class ChatCompletionDelta(Model):
    content: Union[str, None] = None

    def __init__(self, content: Union[str, None]):
        if content is not None:
            self.content = content

    def to_json(self):
        return self.__dict__

class ChatCompletionDeltaChoice(Model):
    def __init__(self, delta: ChatCompletionDelta, finish_reason: Union[str, None]):
        self.delta = delta
        self.finish_reason = finish_reason

    def to_json(self):
        return {
            **self.__dict__,
            "delta": self.delta.to_json()
        }

class Image(Model):
    url: str

    def __init__(self, url: str) -> None:
        self.url = url

class ImagesResponse(Model):
    data: list[Image]

    def __init__(self, data: list) -> None:
        self.data = data