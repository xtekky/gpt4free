from __future__ import annotations

import requests
import json
import base64
from typing import Optional

from ..helper import filter_none
from ...typing import AsyncResult, Messages, MediaListType
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
    default_model = "claude-sonnet-4-20250514"
    
    # Updated Claude 4 models with current versions<!--citation:1--><!--citation:2--><!--citation:3--><!--citation:4--><!--citation:5-->
    models = [
        # Claude 4 models
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250522",
        
        # Claude 3.7 model
        "claude-3-7-sonnet-20250219",
        
        # Claude 3.5 models
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        
        # Legacy Claude 3 models (still supported)
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        
        # Latest aliases
        "claude-opus-4-1-latest",
        "claude-sonnet-4-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "claude-3-opus-latest",
    ]
    
    models_aliases = {
        # Claude 4 aliases
        "claude-4-opus": "claude-opus-4-1-20250805",
        "claude-4.1-opus": "claude-opus-4-1-20250805", 
        "claude-4-sonnet": "claude-sonnet-4-20250514",
        "claude-opus-4": "claude-opus-4-20250522",
        
        # Claude 3.x aliases
        "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",
        "claude-3-opus": "claude-3-opus-20240229",
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
        media: MediaListType = None,
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
        beta_headers: Optional[list] = None,
        extra_body: dict = {},
        **kwargs
    ) -> AsyncResult:
        if api_key is None:
            raise MissingAuthError('Add a "api_key"')

        # Handle image inputs
        if media is not None:
            insert_images = []
            for image, _ in media:
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
        
        # Extract system messages
        system = "\n".join([message["content"] for message in messages if message.get("role") == "system"])
        if system:
            messages = [message for message in messages if message.get("role") != "system"]
        else:
            system = None

        # Get model name
        model_name = cls.get_model(model, api_key=api_key)
        
        # Special handling for Opus 4.1 parameters<!--citation:6-->
        if "opus-4-1" in model_name:
            # Opus 4.1 doesn't allow both temperature and top_p
            if temperature is not None and top_p is not None:
                top_p = None  # Prefer temperature over top_p

        async with StreamSession(
            proxy=proxy,
            headers=cls.get_headers(stream, api_key, headers, beta_headers),
            timeout=timeout,
            impersonate=impersonate,
        ) as session:
            data = filter_none(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                stop_sequences=stop,
                system=system,
                stream=stream,
                tools=tools,
                **extra_body
            )
            async with session.post(f"{cls.api_base}/messages", json=data) as response:
                await raise_for_status(response)
                if not stream:
                    data = await response.json()
                    cls.raise_error(data)
                    tool_calls = []
                    if "type" in data and data["type"] == "message":
                        for content in data["content"]:
                            if content["type"] == "text":
                                yield content["text"]
                            elif content["type"] == "tool_use":
                                tool_calls.append({
                                    "id": content["id"],
                                    "type": "function",
                                    "function": { "name": content["name"], "arguments": json.dumps(content["input"]) }
                                })
                        if tool_calls:
                            yield ToolCalls(tool_calls)
                        if data.get("stop_reason") == "end_turn":
                            yield FinishReason("stop")
                        elif data.get("stop_reason") == "max_tokens":
                            yield FinishReason("length")
                        if "usage" in data:
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
                                elif data["type"] == "content_block_delta":
                                    if content_block and content_block["type"] == "text":
                                        yield data["delta"]["text"]
                                    elif content_block and content_block["type"] == "tool_use":
                                        partial_json.append(data["delta"]["partial_json"])
                                elif data["type"] == "message_delta":
                                    if data["delta"].get("stop_reason") == "end_turn":
                                        yield FinishReason("stop")
                                    elif data["delta"].get("stop_reason") == "max_tokens":
                                        yield FinishReason("length")
                                    if "usage" in data:
                                        yield Usage(**data["usage"])
                                elif data["type"] == "content_block_stop":
                                    if content_block and content_block["type"] == "tool_use":
                                        tool_calls.append({
                                            "id": content_block["id"],
                                            "type": "function",
                                            "function": { 
                                                "name": content_block["name"], 
                                                "arguments": "".join(partial_json)
                                            }
                                        })
                                        partial_json = []
                                    content_block = None
                    if tool_calls:
                        yield ToolCalls(tool_calls)

    @classmethod
    def get_headers(cls, stream: bool, api_key: str = None, headers: dict = None, beta_headers: Optional[list] = None) -> dict:
        result = {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            **(
                {"x-api-key": api_key}
                if api_key is not None else {}
            ),
            "anthropic-version": "2023-06-01",
            **({} if headers is None else headers)
        }
        
        # Add beta headers for special features<!--citation:6--><!--citation:7--><!--citation:8-->
        if beta_headers:
            result["anthropic-beta"] = ",".join(beta_headers)
        
        return result
