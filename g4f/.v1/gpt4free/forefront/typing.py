from typing import Any, List

from pydantic import BaseModel


class Choice(BaseModel):
    text: str
    index: int
    logprobs: Any
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ForeFrontResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    text: str


class AccountData(BaseModel):
    token: str
    user_id: str
    session_id: str
