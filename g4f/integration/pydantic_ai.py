from __future__ import annotations

from typing import Optional
from functools import partial
from dataclasses import dataclass, field

from pydantic_ai import ModelResponsePart, ThinkingPart, ToolCallPart
from pydantic_ai.models import Model, ModelResponse, KnownModelName, infer_model
from pydantic_ai.usage import RequestUsage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openai import OpenAISystemPromptRole, _now_utc, split_content_into_text_and_thinking, replace

import pydantic_ai.models.openai
pydantic_ai.models.openai.NOT_GIVEN = None

from ..client import AsyncClient, ChatCompletion

@dataclass(init=False)
class AIModel(OpenAIChatModel):
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
        system: str | None = 'g4f',
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
        self._provider = getattr(provider, '__name__', provider)
        self.client = AsyncClient(provider=provider, **kwargs)
        self.system_prompt_role = system_prompt_role
        self._system = system

    def name(self) -> str:
        if self._provider:
            return f'g4f:{self._provider}:{self._model_name}'
        return f'g4f:{self._model_name}'
    
    def _process_response(self, response: ChatCompletion | str) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        choice = response.choices[0]
        items: list[ModelResponsePart] = []

        if reasoning := getattr(choice.message, 'reasoning', None):
            items.append(ThinkingPart(id='reasoning', content=reasoning, provider_name=self.system))

        if choice.message.content:
            items.extend(
                (replace(part, id='content', provider_name=self.system) if isinstance(part, ThinkingPart) else part)
                for part in split_content_into_text_and_thinking(choice.message.content, self.profile.thinking_tags)
            )
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                items.append(ToolCallPart(c.function.name, c.function.arguments, tool_call_id=c.id))
        usage = RequestUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

        return ModelResponse(
            parts=items,
            usage=usage,
            model_name=response.model,
            timestamp=_now_utc(),
            provider_details=None,
            provider_response_id=response.id,
            provider_name=self._provider,
            finish_reason=choice.finish_reason,
        )

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