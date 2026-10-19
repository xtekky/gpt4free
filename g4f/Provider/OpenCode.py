from __future__ import annotations

import time

from .template import OpenaiTemplate


class OpenCode(OpenaiTemplate):
    """OpenCode Zen, restricted to the models available on the free tier."""

    label = "OpenCode Zen"
    url = "https://opencode.ai"
    base_url = "https://opencode.ai/zen/v1"
    working = True
    active_by_default = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = "space-bunny-free"

    # Only models that can be used without a paid OpenCode subscription.
    models = [
        "beam-free",
        "mimo-v2.6-flash-free",
        "nemotron-3.5-super-vl-free",
        "nemotron-3.5-lightning-free",
        "jev-1.13-free",
        "longcat-2.5-preview-free",
        "deepseek-v4.1-flash-free",
        "glm-5.3-flash-free",
        "space-bunny-free",
    ]
    fallback_models = models

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "opencode/1.18.31 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14",
        "x-opencode-client": "cli",
        "x-opencode-project": "global",
    }

    @staticmethod
    def is_free_model(model: str) -> bool:
        return model.endswith("-free") or model.endswith(":free")

    @classmethod
    def get_headers(
        cls, stream: bool, api_key: str = None, headers: dict = None
    ) -> dict:
        headers = super().get_headers(stream, api_key, headers)
        timestamp = int(time.time() * 1000)
        headers.setdefault("x-opencode-session", f"ses_{timestamp}")
        headers.setdefault("x-opencode-request", f"msg_{timestamp}")
        return headers

    @classmethod
    def get_models(cls, api_key: str = None, **kwargs) -> list[str]:
        models = super().get_models(api_key=api_key, **kwargs)
        free_models = [model for model in models if cls.is_free_model(model)]
        if free_models:
            cls.models = free_models
        return cls.models
