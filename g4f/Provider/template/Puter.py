from __future__ import annotations

from ...typing import Messages, AsyncResult
from ...providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin

class Puter(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Puter.js AI (live)"
    working = True
    models = [
        {"group": "ChatGPT", "models": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4.5-preview"
        ]},
        {"group": "O Models", "models": [
            "o1",
            "o1-mini",
            "o1-pro",
            "o3",
            "o3-mini",
            "o4-mini"
        ]},
        {"group": "Anthropic Claude", "models": [
            "claude-3-7-sonnet",
            "claude-3-5-sonnet"
        ]},
        {"group": "Deepseek", "models": [
            "deepseek-chat",
            "deepseek-reasoner"
        ]},
        {"group": "Google Gemini", "models": [
            "gemini-2.0-flash",
            "gemini-1.5-flash"
        ]},
        {"group": "Meta Llama", "models": [
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama--70B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        ]},
        {"group": "Other Models", "models": [
            "mistral-large-latest",
            "pixtral-large-latest",
            "codestral-latest",
            "google/gemma-2-27b-it",
            "grok-beta"
        ]}
    ]

    @classmethod
    def get_grouped_models(cls) -> dict[str, list[str]]:
        return cls.models

    def get_models(cls) -> list[str]:
        models = []
        for model in cls.models:
            if "models" in model:
                models.extend(model["models"])
            else:
                models.append(model)
        return models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        **kwargs
    ) -> AsyncResult:
        raise NotImplementedError()