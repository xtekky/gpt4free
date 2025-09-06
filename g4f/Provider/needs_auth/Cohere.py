from __future__ import annotations

import requests

from ..helper import filter_none
from ...typing import AsyncResult, Messages
from ...requests import StreamSession, raise_for_status, sse_stream
from ...providers.response import FinishReason, Usage
from ...errors import MissingAuthError
from ...tools.run_tools import AuthManager
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ... import debug

class Cohere(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Cohere API"
    url = "https://cohere.com"
    login_url = "https://dashboard.cohere.com/api-keys"
    api_endpoint = "https://api.cohere.ai/v2/chat"
    working = True
    active_by_default = True
    needs_auth = True
    models_needs_auth = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "command-r-plus"

    @classmethod
    def get_models(cls, api_key: str = None, **kwargs):
        if not cls.models:
            if not api_key:
                api_key = AuthManager.load_api_key(cls)
            url = "https://api.cohere.com/v1/models?page_size=500&endpoint=chat"
            models = requests.get(url, headers={"Authorization": f"Bearer {api_key}" }).json().get("models", [])
            if models:
                cls.live += 1
            cls.models = [model.get("name") for model in models if "chat" in model.get("endpoints")]
            cls.vision_models = {model.get("name") for model in models if model.get("supports_vision")}
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        api_key: str = None,
        temperature: float = None,
        max_tokens: int = None,
        top_k: int = None,
        top_p: float = None,
        stop: list[str] = None,
        stream: bool = True,
        headers: dict = None,
        impersonate: str = None,
        **kwargs
    ) -> AsyncResult:
        if api_key is None:
            raise MissingAuthError('Add a "api_key"')

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
                k=top_k,
                p=top_p,
                stop_sequences=stop,
                stream=stream,
            )
            async with session.post(cls.api_endpoint, json=data) as response:
                await raise_for_status(response)
                
                if not stream:
                    data = await response.json()
                    cls.raise_error(data)
                    if "text" in data:
                        yield data["text"]
                    if "finish_reason" in data:
                        if data["finish_reason"] == "COMPLETE":
                            yield FinishReason("stop")
                        elif data["finish_reason"] == "MAX_TOKENS":
                            yield FinishReason("length")
                    if "usage" in data:
                        tokens = data.get("usage", {}).get("tokens", {})
                        yield Usage(
                            prompt_tokens=tokens.get("input_tokens"),
                            completion_tokens=tokens.get("output_tokens"),
                            total_tokens=tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0),
                            billed_units=data.get("usage", {}).get("billed_units")
                        )
                else:
                    async for data in sse_stream(response):
                        cls.raise_error(data)
                        if "type" in data:
                            if data["type"] == "content-delta":
                                yield data.get("delta", {}).get("message", {}).get("content", {}).get("text")
                            elif data["type"] == "message-end":
                                delta = data.get("delta", {})
                                if "finish_reason" in delta:
                                    if delta["finish_reason"] == "COMPLETE":
                                        yield FinishReason("stop")
                                    elif delta["finish_reason"] == "MAX_TOKENS":
                                        yield FinishReason("length")
                                if "usage" in delta:
                                    tokens = delta.get("usage", {}).get("tokens", {})
                                    yield Usage(
                                        prompt_tokens=tokens.get("input_tokens"),
                                        completion_tokens=tokens.get("output_tokens"),
                                        total_tokens=tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0),
                                        billed_units=delta.get("usage", {}).get("billed_units")
                                    )

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

    @classmethod
    def raise_error(cls, data: dict):
        if "error" in data:
            raise RuntimeError(f"Cohere API Error: {data['error']}")