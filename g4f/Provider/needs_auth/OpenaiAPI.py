from __future__ import annotations

import json
import requests

from ..helper import filter_none
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...typing import Union, Optional, AsyncResult, Messages, ImagesType
from ...requests import StreamSession, raise_for_status
from ...providers.response import FinishReason, ToolCalls, Usage
from ...errors import MissingAuthError, ResponseError
from ...image import to_data_uri
from ... import debug

class OpenaiAPI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "OpenAI API"
    url = "https://platform.openai.com"
    api_base = "https://api.openai.com/v1"
    working = True
    needs_auth = True
    supports_message_history = True
    supports_system_message = True
    default_model = ""
    fallback_models = []

    @classmethod
    def get_models(cls, api_key: str = None, api_base: str = None) -> list[str]:
        if not cls.models:
            try:
                headers = {}
                if api_base is None:
                    api_base = cls.api_base
                if api_key is not None:
                    headers["authorization"] = f"Bearer {api_key}"
                response = requests.get(f"{api_base}/models", headers=headers)
                raise_for_status(response)
                data = response.json()
                cls.models = [model.get("id") for model in data.get("data")]
                cls.models.sort()
            except Exception as e:
                debug.log(e)
                cls.models = cls.fallback_models
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        images: ImagesType = None,
        api_key: str = None,
        api_endpoint: str = None,
        api_base: str = None,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        stop: Union[str, list[str]] = None,
        stream: bool = False,
        headers: dict = None,
        impersonate: str = None,
        tools: Optional[list] = None,
        extra_data: dict = {},
        **kwargs
    ) -> AsyncResult:
        if cls.needs_auth and api_key is None:
            raise MissingAuthError('Add a "api_key"')
        if api_base is None:
            api_base = cls.api_base
        if images is not None:
            if not model and hasattr(cls, "default_vision_model"):
                model = cls.default_vision_model
            messages[-1]["content"] = [
                *[{
                    "type": "image_url",
                    "image_url": {"url": to_data_uri(image)}
                } for image, _ in images],
                {
                    "type": "text",
                    "text": messages[-1]["content"]
                }
            ]
        async with StreamSession(
            proxy=proxy,
            headers=cls.get_headers(stream, api_key, headers),
            timeout=timeout,
            impersonate=impersonate,
        ) as session:
            data = filter_none(
                messages=messages,
                model=cls.get_model(model, api_key=api_key, api_base=api_base),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                stream=stream,
                tools=tools,
                **extra_data
            )
            if api_endpoint is None:
                api_endpoint = f"{api_base.rstrip('/')}/chat/completions"
            async with session.post(api_endpoint, json=data) as response:
                await raise_for_status(response)
                if not stream:
                    data = await response.json()
                    cls.raise_error(data)
                    choice = data["choices"][0]
                    if "content" in choice["message"] and choice["message"]["content"]:
                        yield choice["message"]["content"].strip()
                    elif "tool_calls" in choice["message"]:
                        yield ToolCalls(choice["message"]["tool_calls"])
                    if "usage" in data:
                        yield Usage(**data["usage"])
                    finish = cls.read_finish_reason(choice)
                    if finish is not None:
                        yield finish
                else:
                    first = True
                    async for line in response.iter_lines():
                        if line.startswith(b"data: "):
                            chunk = line[6:]
                            if chunk == b"[DONE]":
                                break
                            data = json.loads(chunk)
                            cls.raise_error(data)
                            choice = data["choices"][0]
                            if "content" in choice["delta"] and choice["delta"]["content"]:
                                delta = choice["delta"]["content"]
                                if first:
                                    delta = delta.lstrip()
                                if delta:
                                    first = False
                                    yield delta
                            finish = cls.read_finish_reason(choice)
                            if finish is not None:
                                yield finish

    @staticmethod
    def read_finish_reason(choice: dict) -> Optional[FinishReason]:
        if "finish_reason" in choice and choice["finish_reason"] is not None:
            return FinishReason(choice["finish_reason"])

    @staticmethod
    def raise_error(data: dict):
        if "error_message" in data:
            raise ResponseError(data["error_message"])
        elif "error" in data:
            if "code" in data["error"]:
                raise ResponseError(f'Error {data["error"]["code"]}: {data["error"]["message"]}')
            elif "message" in data["error"]:
                raise ResponseError(data["error"]["message"])
            else:
                raise ResponseError(data["error"])

    @classmethod
    def get_headers(cls, stream: bool, api_key: str = None, headers: dict = None) -> dict:
        return {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            **(
                {"Authorization": f"Bearer {api_key}"}
                if api_key is not None else {}
            ),
            **({} if headers is None else headers)
        }