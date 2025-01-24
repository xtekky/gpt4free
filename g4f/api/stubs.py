from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Union, Optional
try:
    from typing import Annotated
except ImportError:
    class Annotated:
        pass
from g4f.typing import Messages

class ChatCompletionsConfig(BaseModel):
    messages: Messages = Field(examples=[[{"role": "system", "content": ""}, {"role": "user", "content": ""}]])
    model: str = Field(default="")
    provider: Optional[str] = None
    stream: bool = False
    image: Optional[str] = None
    image_name: Optional[str] = None
    images: Optional[list[tuple[str, str]]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Union[list[str], str, None] = None
    api_key: Optional[str] = None
    api_base: str = None
    web_search: Optional[bool] = None
    proxy: Optional[str] = None
    conversation_id: Optional[str] = None
    conversation: Optional[dict] = None
    history_disabled: Optional[bool] = None
    auto_continue: Optional[bool] = None
    timeout: Optional[int] = None
    tool_calls: list = Field(default=[], examples=[[
		{
			"function": {
				"arguments": {"query":"search query", "max_results":5, "max_words": 2500, "backend": "auto", "add_text": True, "timeout": 5},
				"name": "search_tool"
			},
			"type": "function"
		}
	]])
    tools: list = None
    response_format: Optional[dict] = None

class ImageGenerationConfig(BaseModel):
    prompt: str
    model: Optional[str] = None
    provider: Optional[str] = None
    response_format: Optional[str] = None
    api_key: Optional[str] = None
    proxy: Optional[str] = None

class ProviderResponseModel(BaseModel):
    id: str
    object: str = "provider"
    created: int
    url: Optional[str]
    label: Optional[str]

class ProviderResponseDetailModel(ProviderResponseModel):
    models: list[str]
    image_models: list[str]
    vision_models: list[str]
    params: list[str]

class ModelResponseModel(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: Optional[str]

class UploadResponseModel(BaseModel):
    bucket_id: str
    url: str

class ErrorResponseModel(BaseModel):
    error: ErrorResponseMessageModel
    model: Optional[str] = None
    provider: Optional[str] = None

class ErrorResponseMessageModel(BaseModel):
    message: str

class FileResponseModel(BaseModel):
    filename: str