from __future__ import annotations

from typing import Optional, List, Dict, Any
from time import time

from .helper import filter_none

try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel():
        @classmethod
        def model_construct(cls, **data):
            new = cls()
            for key, value in data.items():
                setattr(new, key, value)
            return new
    class Field():
        def __init__(self, **config):
            pass

class BaseModel(BaseModel):
    @classmethod
    def model_construct(cls, **data):
        if hasattr(super(), "model_construct"):
            return super().model_construct(**data)
        return cls.construct(**data)

class TokenDetails(BaseModel):
    cached_tokens: int

class UsageModel(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: TokenDetails
    completion_tokens_details: TokenDetails

    @classmethod
    def model_construct(cls, prompt_tokens=0, completion_tokens=0, total_tokens=0, prompt_tokens_details=None, completion_tokens_details=None, **kwargs):
        return super().model_construct(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=TokenDetails.model_construct(**prompt_tokens_details if prompt_tokens_details else {"cached_tokens": 0}),
            completion_tokens_details=TokenDetails.model_construct(**completion_tokens_details if completion_tokens_details else {}),
            **kwargs
        )

class ToolFunctionModel(BaseModel):
    name: str
    arguments: str

class ToolCallModel(BaseModel):
    id: str
    type: str
    function: ToolFunctionModel

    @classmethod
    def model_construct(cls, function=None, **kwargs):
        return super().model_construct(
            **kwargs,
            function=ToolFunctionModel.model_construct(**function),
        )

class ChatCompletionChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    provider: Optional[str]
    choices: List[ChatCompletionDeltaChoice]
    usage: UsageModel

    @classmethod
    def model_construct(
        cls,
        content: str,
        finish_reason: str,
        completion_id: str = None,
        created: int = None,
        usage: UsageModel = None
    ):
        return super().model_construct(
            id=f"chatcmpl-{completion_id}" if completion_id else None,
            object="chat.completion.cunk",
            created=created,
            model=None,
            provider=None,
            choices=[ChatCompletionDeltaChoice.model_construct(
                ChatCompletionDelta.model_construct(content),
                finish_reason
            )],
            **filter_none(usage=usage)
        )

class ChatCompletionMessage(BaseModel):
    role: str
    content: str
    tool_calls: list[ToolCallModel] = None

    @classmethod
    def model_construct(cls, content: str, tool_calls: list = None):
        return super().model_construct(role="assistant", content=content, **filter_none(tool_calls=tool_calls))

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str

    @classmethod
    def model_construct(cls, message: ChatCompletionMessage, finish_reason: str):
        return super().model_construct(index=0, message=message, finish_reason=finish_reason)

class ChatCompletion(BaseModel):
    id: str
    object: str
    created: int
    model: str
    provider: Optional[str]
    choices: List[ChatCompletionChoice]
    usage: UsageModel

    @classmethod
    def model_construct(
        cls,
        content: str,
        finish_reason: str,
        completion_id: str = None,
        created: int = None,
        tool_calls: list[ToolCallModel] = None,
        usage: UsageModel = None
    ):
        return super().model_construct(
            id=f"chatcmpl-{completion_id}" if completion_id else None,
            object="chat.completion",
            created=created,
            model=None,
            provider=None,
            choices=[ChatCompletionChoice.model_construct(
                ChatCompletionMessage.model_construct(content, tool_calls),
                finish_reason,
            )],
            **filter_none(usage=usage)
        )

class ChatCompletionDelta(BaseModel):
    role: str
    content: str

    @classmethod
    def model_construct(cls, content: Optional[str]):
        return super().model_construct(role="assistant", content=content)

class ChatCompletionDeltaChoice(BaseModel):
    index: int
    delta: ChatCompletionDelta
    finish_reason: Optional[str]

    @classmethod
    def model_construct(cls, delta: ChatCompletionDelta, finish_reason: Optional[str]):
        return super().model_construct(index=0, delta=delta, finish_reason=finish_reason)

class Image(BaseModel):
    url: Optional[str]
    b64_json: Optional[str]
    revised_prompt: Optional[str]

    @classmethod
    def model_construct(cls, url: str = None, b64_json: str = None, revised_prompt: str = None):
        return super().model_construct(**filter_none(
            url=url,
            b64_json=b64_json,
            revised_prompt=revised_prompt
        ))

class ImagesResponse(BaseModel):
    data: List[Image]
    model: str
    provider: str
    created: int

    @classmethod
    def model_construct(cls, data: List[Image], created: int = None, model: str = None, provider: str = None):
        if created is None:
            created = int(time())
        return super().model_construct(
            data=data,
            model=model,
            provider=provider,
            created=created
        )