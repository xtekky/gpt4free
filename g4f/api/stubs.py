from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from typing import Union, Optional

from ..typing import Messages

class RequestConfig(BaseModel):
    model: str = Field(default="")
    provider: Optional[str] = None
    media: Optional[list[tuple[str, str]]] = None
    modalities: Optional[list[str]] = None
    temperature: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Union[list[str], str, None] = None
    api_key: Optional[Union[str, dict[str, str]]] = None
    api_base: str = None
    web_search: Optional[bool] = None
    proxy: Optional[str] = None
    conversation: Optional[dict] = None
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
    reasoning_effort: Optional[str] = None
    logit_bias: Optional[dict] = None
    modalities: Optional[list[str]] = None
    audio: Optional[dict] = None
    response_format: Optional[dict] = None
    download_media: bool = True
    extra_body: Optional[dict] = None

class ChatCompletionsConfig(RequestConfig):
    messages: Messages = Field(examples=[[{"role": "system", "content": ""}, {"role": "user", "content": ""}]])
    stream: bool = False
    image: Optional[str] = None
    image_name: Optional[str] = None
    images: Optional[list[tuple[str, str]]] = None
    tools: list = None
    parallel_tool_calls: bool = None
    tool_choice: Optional[str] = None
    conversation_id: Optional[str] = None

class ResponsesConfig(RequestConfig):
    input: Union[Messages, str]

class ImageGenerationConfig(BaseModel):
    prompt: str
    model: Optional[str] = None
    provider: Optional[str] = None
    response_format: Optional[str] = None
    api_key: Optional[str] = None
    proxy: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    seed: Optional[int] = None
    guidance_scale: Optional[int] = None
    aspect_ratio: Optional[str] = None
    n: Optional[int] = None
    negative_prompt: Optional[str] = None
    resolution: Optional[str] = None
    audio: Optional[dict] = None
    download_media: bool = True


    @model_validator(mode='before')
    def parse_size(cls, values):
        if values.get('width') is not None and values.get('height') is not None:
            return values

        size = values.get('size')
        if size:
            try:
                width, height = map(int, size.split('x'))
                values['width'] = width
                values['height'] = height
            except (ValueError, AttributeError): pass  # If the format is incorrect, we simply ignore it.
        return values

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

class TranscriptionResponseModel(BaseModel):
    text: str
    model: str
    provider: str

class AudioSpeechConfig(BaseModel):
    input: str
    model: Optional[str] = None
    provider: Optional[str] = None
    voice: Optional[str] = None
    instrcutions: str = "Speech this text in a natural way."
    response_format: Optional[str] = None
    language: Optional[str] = None