from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ..typing import AsyncResult, Messages, Cookies
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin, get_running_loop
from ..requests import Session, StreamSession, get_args_from_nodriver, raise_for_status, merge_cookies
from ..requests import DEFAULT_HEADERS, has_nodriver, has_curl_cffi
from ..providers.response import FinishReason
from ..cookies import get_cookies_dir
from ..errors import ResponseStatusError, ModelNotFoundError

class Cloudflare(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Cloudflare AI"
    url = "https://playground.ai.cloudflare.com"
    working = True
    use_nodriver = True
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
    _args: dict = None

    @classmethod
    def get_cache_file(cls) -> Path:
        return Path(get_cookies_dir()) / f"auth_{cls.parent if hasattr(cls, 'parent') else cls.__name__}.json"

    @classmethod
    def get_models(cls) -> str:
        if not cls.models:
            if cls._args is None:
                if has_nodriver:
                    get_running_loop(check_nested=True)
                    args = get_args_from_nodriver(cls.url)
                    cls._args = asyncio.run(args)
                elif not has_curl_cffi:
                    return cls.models
                else:
                    cls._args = {"headers": DEFAULT_HEADERS, "cookies": {}}
            with Session(**cls._args) as session:
                response = session.get(cls.models_url)
                cls._args["cookies"] = merge_cookies(cls._args["cookies"], response)
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
        cache_file = cls.get_cache_file()
        if cls._args is None:
            if cache_file.exists():
                with cache_file.open("r") as f:
                    cls._args = json.load(f)
            if has_nodriver:
                cls._args = await get_args_from_nodriver(cls.url, proxy, timeout, cookies)
            else:
                cls._args = {"headers": DEFAULT_HEADERS, "cookies": {}}
        try:
            model = cls.get_model(model)
        except ModelNotFoundError:
            pass
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
                except ResponseStatusError:
                    cls._args = None
                    if cache_file.exists():
                        cache_file.unlink()
                    raise
                reason = None
                async for line in response.iter_lines():
                    if line.startswith(b'data: '):
                        if line == b'data: [DONE]':
                            break
                        try:
                            content = json.loads(line[6:].decode())
                            if content.get("response") and content.get("response") != '</s>':
                                yield content['response']
                                reason = "max_tokens"
                            elif content.get("response") == '':
                                reason = "stop"
                        except Exception:
                            continue
                if reason is not None:
                    yield FinishReason(reason)

                with cache_file.open("w") as f:
                    json.dump(cls._args, f)