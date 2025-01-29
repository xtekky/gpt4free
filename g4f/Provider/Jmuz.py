from __future__ import annotations

from ..typing import AsyncResult, Messages
from .template import OpenaiTemplate

class Jmuz(OpenaiTemplate):
    url = "https://discord.gg/Ew6JzjA2NR"
    api_base = "https://jmuz.me/gpt/api/v2"
    api_key = "prod"
    working = True
    supports_system_message = False

    default_model = "gpt-4o"
    model_aliases = {
        "qwq-32b": "qwq-32b-preview",
        "gemini-1.5-flash": "gemini-flash",
        "gemini-1.5-pro": "gemini-pro",
        "gemini-2.0-flash-thinking": "gemini-thinking",
        "deepseek-chat": "deepseek-v3",
    }

    @classmethod
    def get_models(cls, **kwargs):
        if not cls.models:
            cls.models = super().get_models(api_key=cls.api_key, api_base=cls.api_base)
        return cls.models

    @classmethod
    async def create_async_generator(
            cls,
            model: str,
            messages: Messages,
            stream: bool = True,
            api_key: str = None, # Remove api_key from kwargs
            **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        headers = {
            "Authorization": f"Bearer {cls.api_key}",
            "Content-Type": "application/json",
            "accept": "*/*",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
        }

        started = False
        buffer = ""
        async for chunk in super().create_async_generator(
            model=model,
            messages=messages,
            api_base=cls.api_base,
            api_key=cls.api_key,
            stream=cls.supports_stream,
            headers=headers,
            **kwargs
        ):
            if isinstance(chunk, str):
                buffer += chunk
                if "Join for free".startswith(buffer) or buffer.startswith("Join for free"):
                    if buffer.endswith("\n"):
                        buffer = ""
                    continue
                if "https://discord.gg/".startswith(buffer) or "https://discord.gg/" in buffer:
                    if "..." in buffer:
                        buffer = ""
                    continue
                if "o1-preview".startswith(buffer) or buffer.startswith("o1-preview"):
                    if "\n" in buffer:
                        buffer = ""
                    continue
                if not started:
                    buffer = buffer.lstrip()
                if buffer:
                    started = True
                    yield buffer
                    buffer = ""
            else:
                yield chunk
