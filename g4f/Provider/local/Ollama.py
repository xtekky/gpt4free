from __future__ import annotations

import json
import requests
import os

from ..needs_auth.OpenaiAPI import OpenaiAPI
from ...requests import StreamSession, raise_for_status
from ...providers.response import Usage, Reasoning
from ...tools.run_tools import AuthManager
from ...typing import AsyncResult, Messages

class Ollama(OpenaiAPI):
    label = "Ollama 🦙"
    url = "https://ollama.com"
    login_url = "https://ollama.com/settings/keys"
    api_endpoint = "https://ollama.com/api/chat"
    needs_auth = False
    working = True
    active_by_default = True
    local_models: list[str] = []
    model_aliases = {
        "gpt-oss-120b": "gpt-oss:120b",
        "gpt-oss-20b": "gpt-oss:20b"
    }

    @classmethod
    def get_models(cls, api_key: str = None, api_base: str = None, **kwargs):
        if not cls.models:
            cls.models = []
            if not api_key:
                api_key = AuthManager.load_api_key(cls)
            if api_key:
                models = requests.get("https://ollama.com/api/tags", {"headers": {"Authorization": f"Bearer {api_key}"}}).json()["models"]
                cls.models = [model["name"] for model in models]
            if api_base is None:
                host = os.getenv("OLLAMA_HOST", "127.0.0.1")
                port = os.getenv("OLLAMA_PORT", "11434")
                url = f"http://{host}:{port}/api/tags"
            else:
                url = api_base.replace("/v1", "/api/tags")
            try:
                models = requests.get(url).json()["models"]
            except requests.exceptions.RequestException as e:
                return cls.models
            cls.local_models = [model["name"] for model in models]
            cls.models = cls.models + cls.local_models
            cls.default_model = next(iter(cls.models), None)
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        api_base: str = None,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if api_base is None:
            host = os.getenv("OLLAMA_HOST", "localhost")
            port = os.getenv("OLLAMA_PORT", "11434")
            api_base: str = f"http://{host}:{port}/v1"
        if model in cls.local_models or not api_key:
            async for chunk in super().create_async_generator(
                model, messages, api_base=api_base, proxy=proxy, **kwargs
            ):
                yield chunk
        else:
            async with StreamSession(headers={"Authorization": f"Bearer {api_key}"}, proxy=proxy) as session:
                async with session.post(cls.api_endpoint, json={
                    "model": model,
                    "messages": messages,
                }) as response:
                    await raise_for_status(response)
                    last_data = {}
                    async for chunk in response.iter_lines():
                        data = json.loads(chunk)
                        last_data = data
                        thinking = data.get("message", {}).get("thinking", "")
                        if thinking:
                            yield Reasoning(thinking)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                    yield Usage(
                        prompt_tokens=last_data.get("prompt_eval_count", 0),
                        completion_tokens=last_data.get("eval_count", 0),
                        total_tokens=last_data.get("prompt_eval_count", 0) + last_data.get("eval_count", 0),
                    )