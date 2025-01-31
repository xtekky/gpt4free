from __future__ import annotations

import requests
import json
import base64
from typing import Optional

from ..helper import filter_none
from ...typing import AsyncResult, Messages, ImagesType
from ...requests import StreamSession, raise_for_status
from ...providers.response import FinishReason, ToolCalls, Usage
from ...errors import MissingAuthError
from ...image import to_bytes, is_accepted_format
from .OpenaiAPI import OpenaiAPI

class Anthropic(OpenaiAPI):
    label = "Anthropic API"
    url = "https://console.anthropic.com"
    login_url = "https://console.anthropic.com/settings/keys"
    working = True
    api_base = "https://api.anthropic.com/v1"
    needs_auth = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    default_model = "claude-3-5-sonnet-latest"
    models = [
        default_model,
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-latest",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-latest",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]
    models_aliases = {
        "claude-3.5-sonnet": default_model,
        "claude-3-opus": "claude-3-opus-latest",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
    }

    @classmethod
    def get_models(cls, api_key: str = None, **kwargs):
        if not cls.models:
            url = f"https://api.anthropic.com/v1/models"
            response = requests.get(url, headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            })
            raise_for_status(response)
            models = response.json()
            cls.models = [model["id"] for model in models["data"]]
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
        temperature: float = None,
        max_tokens: int = 4096,
        top_k: int = None,
        top_p: float = None,
        stop: list[str] = None,
        stream: bool = False,
        headers: dict = None,
        impersonate: str = None,
        tools: Optional[list] = None,
        extra_data: dict = {},
        **kwargs
    ) -> AsyncResult:
        if api_key is None:
            raise MissingAuthError('Add a "api_key"')

        if images is not None:
            insert_images = []
            for image, _ in images:
                data = to_bytes(image)
                insert_images.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": is_accepted_format(data),
                        "data": base64.b64encode(data).decode(),
                    }
                })
            messages[-1]["content"] = [
                *insert_images,
                {
                    "type": "text",
                    "text": messages[-1]["content"]
                }
            ]
        system = "\n".join([message["content"] for message in messages if message.get("role") == "system"])
        if system:
            messages = [message for message in messages if message.get("role") != "system"]
        else:
            system = None

        async with StreamSession(
            proxy=proxy,
            headers=cls.get_headers(stream, api_key, headers),
            timeout=timeout,
            impersonate=impersonate,
        ) as session:
            data = filter_none(
                messages=messages,
                model=cls.get_model(model, api_key=api_key),
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                stop_sequences=stop,
                system=system,
                stream=stream,
                tools=tools,
                **extra_data
            )
            async with session.post(f"{cls.api_base}/messages", json=data) as response:
                await raise_for_status(response)
                if not stream:
                    data = await response.json()
                    cls.raise_error(data)
                    if "type" in data and data["type"] == "message":
                        for content in data["content"]:
                            if content["type"] == "text":
                                yield content["text"]
                            elif content["type"] == "tool_use":
                                tool_calls.append({
                                    "id": content["id"],
                                    "type": "function",
                                    "function": { "name": content["name"], "arguments": content["input"] }
                                })
                        if data["stop_reason"] == "end_turn":
                            yield FinishReason("stop")
                        elif data["stop_reason"] == "max_tokens":
                            yield FinishReason("length")
                        yield Usage(**data["usage"])
                else:
                    content_block = None
                    partial_json = []
                    tool_calls = []
                    async for line in response.iter_lines():
                        if line.startswith(b"data: "):
                            chunk = line[6:]
                            if chunk == b"[DONE]":
                                break
                            data = json.loads(chunk)
                            cls.raise_error(data)
                            if "type" in data:
                                if data["type"] == "content_block_start":
                                    content_block = data["content_block"]
                                if content_block is None:
                                    pass # Message start
                                elif data["type"] == "content_block_delta":
                                    if content_block["type"] == "text":
                                        yield data["delta"]["text"]
                                    elif content_block["type"] == "tool_use":
                                        partial_json.append(data["delta"]["partial_json"])
                                elif data["type"] == "message_delta":
                                    if data["delta"]["stop_reason"] == "end_turn":
                                        yield FinishReason("stop")
                                    elif data["delta"]["stop_reason"] == "max_tokens":
                                        yield FinishReason("length")
                                    yield Usage(**data["usage"])
                                elif data["type"] == "content_block_stop":
                                    if content_block["type"] == "tool_use":
                                        tool_calls.append({
                                            "id": content_block["id"],
                                            "type": "function",
                                            "function": { "name": content_block["name"], "arguments": partial_json.join("") }
                                        })
                                        partial_json = []
                    if tool_calls:
                        yield ToolCalls(tool_calls)

    @classmethod
    def get_headers(cls, stream: bool, api_key: str = None, headers: dict = None) -> dict:
        return {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            **(
                {"x-api-key": api_key}
                if api_key is not None else {}
            ),
            "anthropic-version": "2023-06-01",
            **({} if headers is None else headers)
        }