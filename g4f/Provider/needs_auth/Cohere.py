from __future__ import annotations

import json
from typing import Optional

from ..helper import filter_none
from ...typing import AsyncResult, Messages
from ...requests import StreamSession, raise_for_status
from ...providers.response import FinishReason, Usage
from ...errors import MissingAuthError
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...tools.run_tools import AuthManager
from ... import debug

class Cohere(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Cohere API"
    url = "https://cohere.com"
    login_url = "https://dashboard.cohere.com/api-keys"
    api_base = "https://api.cohere.ai/v1"
    working = True
    needs_auth = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "command-r-plus"
    models = [
        default_model,
        "command-r",
        "command",
        "command-nightly",
        "command-light",
        "command-light-nightly",
    ]
    
    model_aliases = {
        "command-r-plus-08-2024": "command-r-plus",
        "command-r-08-2024": "command-r",
    }

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
        stream: bool = False,
        headers: dict = None,
        impersonate: str = None,
        **kwargs
    ) -> AsyncResult:
        if api_key is None:
            api_key = AuthManager.load_api_key(cls)
        if api_key is None:
            raise MissingAuthError('Add a "api_key"')

        # Convert messages to Cohere format
        system_message = None
        chat_history = []
        user_message = None
        
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            
            if role == "system":
                system_message = content
            elif role == "user":
                if user_message is not None:
                    # Previous user message becomes part of chat history
                    chat_history.append({"role": "USER", "message": user_message})
                user_message = content
            elif role == "assistant":
                chat_history.append({"role": "CHATBOT", "message": content})

        # Ensure we have a user message
        if user_message is None:
            raise ValueError("No user message found in the conversation")

        async with StreamSession(
            proxy=proxy,
            headers=cls.get_headers(stream, api_key, headers),
            timeout=timeout,
            impersonate=impersonate,
        ) as session:
            data = filter_none(
                message=user_message,
                model=cls.get_model(model, api_key=api_key),
                temperature=temperature,
                max_tokens=max_tokens,
                k=top_k,
                p=top_p,
                stop_sequences=stop,
                preamble=system_message,
                chat_history=chat_history if chat_history else None,
                stream=stream,
            )
            
            async with session.post(f"{cls.api_base}/chat", json=data) as response:
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
                    if "meta" in data and "tokens" in data["meta"]:
                        yield Usage(
                            prompt_tokens=data["meta"]["tokens"]["input_tokens"],
                            completion_tokens=data["meta"]["tokens"]["output_tokens"],
                            total_tokens=data["meta"]["tokens"]["input_tokens"] + data["meta"]["tokens"]["output_tokens"]
                        )
                else:
                    async for line in response.iter_lines():
                        if line.startswith(b"data: "):
                            chunk = line[6:]
                            if chunk == b"[DONE]":
                                break
                            try:
                                data = json.loads(chunk)
                                cls.raise_error(data)
                                
                                if "event_type" in data:
                                    if data["event_type"] == "text-generation":
                                        if "text" in data:
                                            yield data["text"]
                                    elif data["event_type"] == "stream-end":
                                        if "finish_reason" in data:
                                            if data["finish_reason"] == "COMPLETE":
                                                yield FinishReason("stop")
                                            elif data["finish_reason"] == "MAX_TOKENS":
                                                yield FinishReason("length")
                                        if "meta" in data and "tokens" in data["meta"]:
                                            yield Usage(
                                                prompt_tokens=data["meta"]["tokens"]["input_tokens"],
                                                completion_tokens=data["meta"]["tokens"]["output_tokens"],
                                                total_tokens=data["meta"]["tokens"]["input_tokens"] + data["meta"]["tokens"]["output_tokens"]
                                            )
                            except json.JSONDecodeError:
                                continue

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