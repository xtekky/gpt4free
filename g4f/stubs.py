
from __future__ import annotations

class Model():
    def __getitem__(self, item):
        return getattr(self, item)

class ChatCompletion(Model):
    def __init__(self, content: str, finish_reason: str):
        self.choices = [ChatCompletionChoice(ChatCompletionMessage(content, finish_reason))]

class ChatCompletionChunk(Model):
    def __init__(self, content: str, finish_reason: str):
        self.choices = [ChatCompletionDeltaChoice(ChatCompletionDelta(content, finish_reason))]

class ChatCompletionMessage(Model):
    def __init__(self, content: str, finish_reason: str):
        self.content = content
        self.finish_reason = finish_reason

class ChatCompletionChoice(Model):
    def __init__(self, message: ChatCompletionMessage):
        self.message = message

class ChatCompletionDelta(Model):
    def __init__(self, content: str, finish_reason: str):
        self.content = content
        self.finish_reason = finish_reason

class ChatCompletionDeltaChoice(Model):
    def __init__(self, delta: ChatCompletionDelta):
        self.delta = delta

class Image(Model):
    url: str

    def __init__(self, url: str) -> None:
        self.url = url

class ImagesResponse(Model):
    data: list[Image]

    def __init__(self, data: list) -> None:
        self.data = data