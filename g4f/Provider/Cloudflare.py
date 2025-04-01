from __future__ import annotations

import json

from ..typing import AsyncResult, Messages, Cookies
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..requests import Session, StreamSession, raise_for_status
from ..requests import DEFAULT_HEADERS
from ..providers.response import FinishReason, Usage
from ..errors import ResponseStatusError, ModelNotFoundError

class Cloudflare(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Cloudflare AI"
    url = "https://playground.ai.cloudflare.com"
    working = True
    use_nodriver = False
    api_endpoint = "https://playground.ai.cloudflare.com/api/inference"
    models_url = "https://playground.ai.cloudflare.com/api/models"
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    default_model = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    model_aliases = {       
        "llama-2-7b": "@cf/meta/llama-2-7b-chat-fp16",
        "llama-2-7b": "@cf/meta/llama-2-7b-chat-int8",
        "llama-3-8b": "@cf/meta/llama-3-8b-instruct",
        "llama-3-8b": "@cf/meta/llama-3-8b-instruct-awq",
        "llama-3-8b": "@hf/meta-llama/meta-llama-3-8b-instruct",
        "llama-3.1-8b": "@cf/meta/llama-3.1-8b-instruct-awq",
        "llama-3.1-8b": "@cf/meta/llama-3.1-8b-instruct-fp8",
        "llama-3.2-1b": "@cf/meta/llama-3.2-1b-instruct",
        "qwen-1.5-7b": "@cf/qwen/qwen1.5-7b-chat-awq",
    }
    
    @classmethod
    def get_models(cls) -> str:
        if not cls.models:
            with Session(headers=DEFAULT_HEADERS) as session:
                response = session.get(cls.models_url)
                try:
                    raise_for_status(response)
                except ResponseStatusError:
                    return cls.models
                json_data = response.json()
                cls.models = [model.get("name") for model in json_data.get("models")]
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        max_tokens: int = 2048,
        cookies: Cookies = None,
        timeout: int = 300,
        **kwargs
    ) -> AsyncResult:
        try:
            model = cls.get_model(model)
        except ModelNotFoundError:
            pass
            
        data = {
            "messages": [{
                **message,
                "content": message["content"] if isinstance(message["content"], str) else "",
                "parts": [{"type":"text", "text":message["content"]}] if isinstance(message["content"], str) else message} for message in messages],
            "lora": None,
            "model": model,
            "max_tokens": max_tokens,
            "stream": True,
            "system_message":"You are a helpful assistant",
            "tools":[]
        }
        
        session_kwargs = {"headers": DEFAULT_HEADERS}
        if proxy:
            session_kwargs["proxies"] = {"https": proxy}
            
        async with StreamSession(**session_kwargs) as session:
            async with session.post(
                cls.api_endpoint,
                json=data,
                timeout=timeout
            ) as response:
                try:
                    await raise_for_status(response)
                except ResponseStatusError:
                    raise
                    
                async for line in response.iter_lines():
                    if line.startswith(b'0:'):
                        yield json.loads(line[2:])
                    elif line.startswith(b'e:'):
                        finish = json.loads(line[2:])
                        yield Usage(**finish.get("usage"))
                        yield FinishReason(finish.get("finishReason"))