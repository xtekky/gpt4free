from __future__ import annotations

import json
import time
import requests

from ..helper import filter_none, format_image_prompt
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, RaiseErrorMixin
from ...typing import Union, Optional, AsyncResult, Messages, ImagesType
from ...requests import StreamSession, raise_for_status
from ...providers.response import FinishReason, ToolCalls, Usage, ImageResponse
from ...errors import MissingAuthError, ResponseError
from ...image import to_data_uri
from ... import debug

class OpenaiTemplate(AsyncGeneratorProvider, ProviderModelMixin, RaiseErrorMixin):
    api_base = ""
    supports_message_history = True
    supports_system_message = True
    default_model = ""
    fallback_models = []
    sort_models = True
    ssl = None

    @classmethod
    def get_models(cls, api_key: str = None, api_base: str = None) -> list[str]:
        if not cls.models:
            try:
                headers = {}
                if api_base is None:
                    api_base = cls.api_base
                if api_key is not None:
                    headers["authorization"] = f"Bearer {api_key}"
                response = requests.get(f"{api_base}/models", headers=headers, verify=cls.ssl)
                raise_for_status(response)
                data = response.json()
                data = data.get("data") if isinstance(data, dict) else data
                cls.image_models = [model.get("id") for model in data if model.get("image")]
                cls.models = [model.get("id") for model in data]
                if cls.sort_models:
                    cls.models.sort()
            except Exception as e:
                debug.log(e)
                return cls.fallback_models
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
        prompt: str = None,
        headers: dict = None,
        impersonate: str = None,
        tools: Optional[list] = None,
        extra_data: dict = {},
        **kwargs
    ) -> AsyncResult:
        if cls.needs_auth and api_key is None:
            raise MissingAuthError('Add a "api_key"')
        async with StreamSession(
            proxy=proxy,
            headers=cls.get_headers(stream, api_key, headers),
            timeout=timeout,
            impersonate=impersonate,
        ) as session:
            model = cls.get_model(model, api_key=api_key, api_base=api_base)
            if api_base is None:
                api_base = cls.api_base

            # Proxy for image generation feature
            if model and model in cls.image_models:
                data = {
                    "prompt": format_image_prompt(messages, prompt),
                    "model": model,
                }
                async with session.post(f"{api_base.rstrip('/')}/images/generations", json=data, ssl=cls.ssl) as response:
                    data = await response.json()
                    cls.raise_error(data)
                    await raise_for_status(response)
                    yield ImageResponse([image["url"] for image in data["data"]], prompt)
                return

            if images is not None and messages:
                if not model and hasattr(cls, "default_vision_model"):
                    model = cls.default_vision_model
                last_message = messages[-1].copy()
                last_message["content"] = [
                    *[{
                        "type": "image_url",
                        "image_url": {"url": to_data_uri(image)}
                    } for image, _ in images],
                    {
                        "type": "text",
                        "text": messages[-1]["content"]
                    }
                ]
                messages[-1] = last_message
            data = filter_none(
                messages=messages,
                model=model,
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
            async with session.post(api_endpoint, json=data, ssl=cls.ssl) as response:
                content_type = response.headers.get("content-type", "text/event-stream" if stream else "application/json")
                if content_type.startswith("application/json"):
                    data = await response.json()
                    cls.raise_error(data)
                    await raise_for_status(response)
                    choice = data["choices"][0]
                    if "content" in choice["message"] and choice["message"]["content"]:
                        yield choice["message"]["content"].strip()
                    elif "tool_calls" in choice["message"]:
                        yield ToolCalls(choice["message"]["tool_calls"])
                    if "usage" in data:
                        yield Usage(**data["usage"])
                    if "finish_reason" in choice and choice["finish_reason"] is not None:
                        yield FinishReason(choice["finish_reason"])
                        return
                elif content_type.startswith("text/event-stream"):
                    await raise_for_status(response)
                    first = True
                    is_thinking = 0
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
                            if "usage" in data and data["usage"]:
                                yield Usage(**data["usage"])
                            if "finish_reason" in choice and choice["finish_reason"] is not None:
                                yield FinishReason(choice["finish_reason"])
                                break
                else:
                    await raise_for_status(response)
                    raise ResponseError(f"Not supported content-type: {content_type}")

    @classmethod
    def get_headers(cls, stream: bool, api_key: str = None, headers: dict = None) -> dict:
        return {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            **(
                {"Authorization": f"Bearer {api_key}"}
                if api_key else {}
            ),
            **({} if headers is None else headers)
        }