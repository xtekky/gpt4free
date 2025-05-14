from __future__ import annotations

from typing import Optional
from functools import partial
from dataclasses import dataclass, field

from pydantic_ai.models import Model, KnownModelName, infer_model
from pydantic_ai.models.openai import OpenAIModel, OpenAISystemPromptRole

import pydantic_ai.models.openai
pydantic_ai.models.openai.NOT_GIVEN = None

from ..client import AsyncClient

@dataclass(init=False)
class AIModel(OpenAIModel):
    """A model that uses the G4F API."""

    client: AsyncClient = field(repr=False)
    system_prompt_role: OpenAISystemPromptRole | None = field(default=None)

    _model_name: str = field(repr=False)
    _provider: str = field(repr=False)
    _system: Optional[str] = field(repr=False)

    def __init__(
        self,
        model_name: str,
        provider: str | None = None,
        *,
        system_prompt_role: OpenAISystemPromptRole | None = None,
        system: str | None = 'openai',
        **kwargs
    ):
        """Initialize an AI model.

        Args:
            model_name: The name of the AI model to use. List of model names available
                [here](https://github.com/openai/openai-python/blob/v1.54.3/src/openai/types/chat_model.py#L7)
                (Unfortunately, despite being ask to do so, OpenAI do not provide `.inv` files for their API).
            system_prompt_role: The role to use for the system prompt message. If not provided, defaults to `'system'`.
                In the future, this may be inferred from the model name.
            system: The model provider used, defaults to `openai`. This is for observability purposes, you must
                customize the `base_url` and `api_key` to use a different provider.
        """
        self._model_name = model_name
        self._provider = provider
        self.client = AsyncClient(provider=provider, **kwargs)
        self.system_prompt_role = system_prompt_role
        self._system = system

    def name(self) -> str:
        if self._provider:
            return f'g4f:{self._provider}:{self._model_name}'
        return f'g4f:{self._model_name}'

def new_infer_model(model: Model | KnownModelName, api_key: str = None) -> Model:
    if isinstance(model, Model):
        return model
    if model.startswith("g4f:"):
        model = model[4:]
        if ":" in model:
            provider, model = model.split(":", 1)
            return AIModel(model, provider=provider, api_key=api_key)
        return AIModel(model)
    return infer_model(model)

def patch_infer_model(api_key: str | None = None):
    import pydantic_ai.models

    pydantic_ai.models.infer_model = partial(new_infer_model, api_key=api_key)
    pydantic_ai.models.AIModel = AIModel