from __future__ import annotations

import asyncio
import json
import uuid

from ..typing import AsyncResult, Messages, Cookies
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin, get_running_loop
from ..requests import Session, StreamSession, get_args_from_nodriver, raise_for_status, merge_cookies
from ..errors import ResponseStatusError

class Cloudflare(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Cloudflare AI"
    url = "https://playground.ai.cloudflare.com"
    api_endpoint = "https://playground.ai.cloudflare.com/api/inference"
    models_url = "https://playground.ai.cloudflare.com/api/models"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    default_model = "@cf/meta/llama-3.1-8b-instruct"
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
    _args: dict = None

    @classmethod
    def get_models(cls) -> str:
        if not cls.models:
            if cls._args is None:
                get_running_loop(check_nested=True)
                args = get_args_from_nodriver(cls.url, cookies={
                    '__cf_bm': uuid.uuid4().hex,
                })
                cls._args = asyncio.run(args)
            with Session(**cls._args) as session:
                response = session.get(cls.models_url)
                cls._args["cookies"] = merge_cookies(cls._args["cookies"] , response)
                try:
                    raise_for_status(response)
                except ResponseStatusError as e:
                    cls._args = None
                    raise e
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
        model = cls.get_model(model)
        if cls._args is None:
            cls._args = await get_args_from_nodriver(cls.url, proxy, timeout, cookies)
        data = {
            "messages": messages,
            "lora": None,
            "model": model,
            "max_tokens": max_tokens,
            "stream": True
        }
        async with StreamSession(**cls._args) as session:
            async with session.post(
                cls.api_endpoint,
                json=data,
            ) as response:
                cls._args["cookies"] = merge_cookies(cls._args["cookies"] , response)
                try:
                    await raise_for_status(response)
                except ResponseStatusError as e:
                    cls._args = None
                    raise e
                async for line in response.iter_lines():
                    if line.startswith(b'data: '):
                        if line == b'data: [DONE]':
                            break
                        try:
                            content = json.loads(line[6:].decode())
                            if content.get("response") and content.get("response") != '</s>':
                                yield content['response']
                        except Exception:
                            continue