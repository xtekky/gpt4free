from __future__ import annotations

import asyncio
import json

from ..typing import AsyncResult, Messages, Cookies
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin, get_running_loop
from ..requests import Session, StreamSession, get_args_from_nodriver, raise_for_status, merge_cookies
from ..requests import DEFAULT_HEADERS, has_nodriver, has_curl_cffi
from ..providers.response import FinishReason, Usage
from ..errors import ResponseStatusError, ModelNotFoundError
from .. import debug
from .helper import render_messages

class Cloudflare(AsyncGeneratorProvider, ProviderModelMixin, AuthFileMixin):
    label = "Cloudflare AI"
    url = "https://playground.ai.cloudflare.com"
    working = has_curl_cffi
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
        "llama-4-scout": "@cf/meta/llama-4-scout-17b-16e-instruct",
        "deepseek-math-7b": "@cf/deepseek-ai/deepseek-math-7b-instruct",
        "deepseek-r1-qwen-32b": "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        "falcon-7b": "@cf/tiiuae/falcon-7b-instruct",
        "qwen-1.5-7b": "@cf/qwen/qwen1.5-7b-chat-awq",
        "qwen-2.5-coder": "@cf/qwen/qwen2.5-coder-32b-instruct",
    }
    fallback_models = list(model_aliases.keys())
    _args: dict = None

    @classmethod
    def get_models(cls) -> str:
        def read_models():
            with Session(**cls._args) as session:
                response = session.get(cls.models_url)
                cls._args["cookies"] = merge_cookies(cls._args["cookies"], response)
                raise_for_status(response)
                json_data = response.json()
                def clean_name(name: str) -> str:
                    return name.split("/")[-1].replace(
                        "-instruct", "").replace(
                        "-17b-16e", "").replace(
                        "-chat", "").replace(
                        "-fp8", "").replace(
                        "-fast", "").replace(
                        "-int8", "").replace(
                        "-awq", "").replace(
                        "-qvq", "").replace(
                        "-r1", "").replace(
                        "meta-llama-", "llama-")
                model_map = {clean_name(model.get("name")): model.get("name") for model in json_data.get("models")}
                cls.models = list(model_map.keys())
                cls.model_aliases = {**cls.model_aliases, **model_map}
        if not cls.models:
            try:
                cache_file = cls.get_cache_file()
                if cls._args is None:
                    if cache_file.exists():
                        with cache_file.open("r") as f:
                            cls._args = json.load(f)
                if cls._args is None:
                    cls._args = {"headers": DEFAULT_HEADERS, "cookies": {}}
                read_models()
            except ResponseStatusError as f:
                if has_nodriver:
                    async def nodriver_read_models():
                        try:
                            cls._args = await get_args_from_nodriver(cls.url)
                            read_models()
                        except Exception as e:
                            debug.log(f"Nodriver is not available: {type(e).__name__}: {e}")
                            cls.models = cls.fallback_models
                    get_running_loop(check_nested=True)
                    try:
                        asyncio.run(nodriver_read_models())
                    except RuntimeError:
                        debug.log("Nodriver is not available: RuntimeError")
                        cls.models = cls.fallback_models
                else:
                    cls.models = cls.fallback_models
                    debug.log(f"Nodriver is not installed: {type(f).__name__}: {f}")
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncResult:
        cache_file = cls.get_cache_file()
        if cls._args is None:
            if cache_file.exists():
                with cache_file.open("r") as f:
                    cls._args = json.load(f)
            elif has_nodriver:
                try:
                    cls._args = await get_args_from_nodriver(cls.url, proxy=proxy)
                except (RuntimeError, FileNotFoundError) as e:
                    debug.log(f"Nodriver is not available: {type(e).__name__}: {e}")
                    cls._args = {"headers": DEFAULT_HEADERS, "cookies": {}, "impersonate": "chrome"}
            else:
                cls._args = {"headers": DEFAULT_HEADERS, "cookies": {}, "impersonate": "chrome"}
        try:
            model = cls.get_model(model)
        except ModelNotFoundError:
            pass
        data = {
            "messages": [{
                **message,
                "parts": [{"type":"text", "text": message["content"]}]} for message in render_messages(messages)],
            "lora": None,
            "model": model,
            "max_tokens": max_tokens,
            "stream": True,
            "system_message":"You are a helpful assistant",
            "tools":[]
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
                async for line in response.iter_lines():
                    if line.startswith(b'0:'):
                        yield json.loads(line[2:])
                    elif line.startswith(b'e:'):
                        finish = json.loads(line[2:])
                        yield Usage(**finish.get("usage"))
                        yield FinishReason(finish.get("finishReason"))

        with cache_file.open("w") as f:
            json.dump(cls._args, f)
