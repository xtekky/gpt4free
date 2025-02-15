from __future__ import annotations
from typing import Optional, List, Dict, Any
from time import time
from .helper import filter_none
ToolCalls = Optional[List[Dict[str, Any]]]
Usage = Optional[Dict[str, int]]
try:
    from pydantic import BaseModel, Field
except ImportError:

    class BaseModel:

        @classmethod
        def model_construct(cls, **data):
            """Return the result of model_construct using parameters: cls."""
            new = cls()
            for key, value in data.items():
                setattr(new, key, value)
            return new

    class Field:

        def __init__(self, **config):
            """Compute __init__ given no arguments."""
            pass

class BaseModel(BaseModel):

    @classmethod
    def model_construct(cls, **data):
        """Compute model_construct given cls."""
        if hasattr(super(), 'model_construct'):
            return super().model_construct(**data)
        return cls.construct(**data)

class ChatCompletionChunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    provider: Optional[str]
    choices: List[ChatCompletionDeltaChoice]
    usage: Usage

    @classmethod
    def model_construct(cls, content: str, finish_reason: str, completion_id: str=None, created: int=None, usage: Usage=None):
        """Return the result of model_construct using parameters: cls, content, finish_reason, completion_id, created, usage."""
        return super().model_construct(id=f'chatcmpl-{completion_id}' if completion_id else None, object='chat.completion.cunk', created=created, model=None, provider=None, choices=[ChatCompletionDeltaChoice.model_construct(ChatCompletionDelta.model_construct(content), finish_reason)], **filter_none(usage=usage))

class ChatCompletionMessage(BaseModel):
    role: str
    content: str
    tool_calls: ToolCalls

    @classmethod
    def model_construct(cls, content: str, tool_calls: ToolCalls=None):
        """Compute model_construct given cls, content, tool_calls."""
        return super().model_construct(role='assistant', content=content, **filter_none(tool_calls=tool_calls))

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str

    @classmethod
    def model_construct(cls, message: ChatCompletionMessage, finish_reason: str):
        """Compute model_construct given cls, message, finish_reason."""
        return super().model_construct(index=0, message=message, finish_reason=finish_reason)

class ChatCompletion(BaseModel):
    id: str
    object: str
    created: int
    model: str
    provider: Optional[str]
    choices: List[ChatCompletionChoice]
    usage: Usage = Field(default={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})

    @classmethod
    def model_construct(cls, content: str, finish_reason: str, completion_id: str=None, created: int=None, tool_calls: ToolCalls=None, usage: Usage=None):
        """Process data using model_construct with arguments cls, content, finish_reason, completion_id, created, tool_calls, usage."""
        return super().model_construct(id=f'chatcmpl-{completion_id}' if completion_id else None, object='chat.completion', created=created, model=None, provider=None, choices=[ChatCompletionChoice.model_construct(ChatCompletionMessage.model_construct(content, tool_calls), finish_reason)], **filter_none(usage=usage))

class ChatCompletionDelta(BaseModel):
    role: str
    content: str

    @classmethod
    def model_construct(cls, content: Optional[str]):
        """Return the result of model_construct using parameters: cls, content."""
        return super().model_construct(role='assistant', content=content)

class ChatCompletionDeltaChoice(BaseModel):
    index: int
    delta: ChatCompletionDelta
    finish_reason: Optional[str]

    @classmethod
    def model_construct(cls, delta: ChatCompletionDelta, finish_reason: Optional[str]):
        """Return the result of model_construct using parameters: cls, delta, finish_reason."""
        return super().model_construct(index=0, delta=delta, finish_reason=finish_reason)

class Image(BaseModel):
    url: Optional[str]
    b64_json: Optional[str]
    revised_prompt: Optional[str]

    @classmethod
    def model_construct(cls, url: str=None, b64_json: str=None, revised_prompt: str=None):
        """Execute model_construct with input (cls, url, b64_json, revised_prompt)."""
        return super().model_construct(**filter_none(url=url, b64_json=b64_json, revised_prompt=revised_prompt))

class ImagesResponse(BaseModel):
    data: List[Image]
    model: str
    provider: str
    created: int

    @classmethod
    def model_construct(cls, data: List[Image], created: int=None, model: str=None, provider: str=None):
        """Calculate and return the output of model_construct based on cls, data, created, model, provider."""
        if created is None:
            created = int(time())
        return super().model_construct(data=data, model=model, provider=provider, created=created)