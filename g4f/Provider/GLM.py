from __future__ import annotations

import uuid
import requests

from ..typing import AsyncResult, Messages
from ..providers.response import Usage, Reasoning
from ..requests import StreamSession, raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

class GLM(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://chat.z.ai"
    api_endpoint = "https://chat.z.ai/api/chat/completions"
    working = True
    active_by_default = True
    default_model = "GLM-4.5"
    api_key = None

    @classmethod
    def get_models(cls, **kwargs) -> str:
        if not cls.models:
            response = requests.get(f"{cls.url}/api/v1/auths/")
            cls.api_key = response.json().get("token")
            response = requests.get(f"{cls.url}/api/models", headers={"Authorization": f"Bearer {cls.api_key}"})
            data = response.json().get("data", [])
            cls.model_aliases = {data.get("name"): data.get("id") for data in data}
            cls.models = list(cls.model_aliases.keys())
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        cls.get_models()
        model = cls.get_model(model)
        data = {
            "chat_id": "local",
            "id": str(uuid.uuid4()),
            "stream": True,
            "model": model,
            "messages": messages,
            "params": {},
            "tool_servers": [],
            "features": {
                "enable_thinking": True
            }
        }
        async with StreamSession(
            impersonate="chrome",
            proxy=proxy,
        ) as session:
            async with session.post(
                cls.api_endpoint,
                json=data,
                headers={"Authorization": f"Bearer {cls.api_key}", "x-fe-version": "prod-fe-1.0.57"},
            ) as response:
                await raise_for_status(response)
                usage = None
                async for chunk in response.sse():
                    if chunk.get("type") == "chat:completion":
                        if not usage:
                            usage = chunk.get("data", {}).get("usage")
                            if usage:
                                yield Usage(**usage)
                        if chunk.get("data", {}).get("phase") == "thinking":
                            delta_content = chunk.get("data", {}).get("delta_content")
                            delta_content = delta_content.split("</summary>\n>")[-1] if delta_content else ""
                            if delta_content:
                                yield Reasoning(delta_content)
                        else:
                            edit_content = chunk.get("data", {}).get("edit_content")
                            if edit_content:
                                yield edit_content.split("\n</details>\n")[-1]
                            else:
                                delta_content = chunk.get("data", {}).get("delta_content")
                                if delta_content:
                                    yield delta_content
