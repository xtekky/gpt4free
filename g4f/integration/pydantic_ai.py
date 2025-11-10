from __future__ import annotations

from typing import Optional
from functools import partial
from dataclasses import dataclass, field

from pydantic_ai import ModelResponsePart, ThinkingPart, ToolCallPart
from pydantic_ai.models import Model, ModelResponse, KnownModelName, infer_model
from pydantic_ai.models.openai import OpenAIChatModel, UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAISystemPromptRole, _CHAT_FINISH_REASON_MAP, _map_usage, _now_utc, number_to_datetime, split_content_into_text_and_thinking, replace

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
    
    def _process_response(self, response: ChatCompletion | str) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        # Although the OpenAI SDK claims to return a Pydantic model (`ChatCompletion`) from the chat completions function:
        # * it hasn't actually performed validation (presumably they're creating the model with `model_construct` or something?!)
        # * if the endpoint returns plain text, the return type is a string
        # Thus we validate it fully here.
        if not isinstance(response, ChatCompletion):
            raise UnexpectedModelBehavior('Invalid response from OpenAI chat completions endpoint, expected JSON data')

        if response.created:
            timestamp = number_to_datetime(response.created)
        else:
            timestamp = _now_utc()
            response.created = int(timestamp.timestamp())

        # Workaround for local Ollama which sometimes returns a `None` finish reason.
        if response.choices and (choice := response.choices[0]) and choice.finish_reason is None:  # pyright: ignore[reportUnnecessaryComparison]
            choice.finish_reason = 'stop'

        choice = response.choices[0]
        items: list[ModelResponsePart] = []

        # The `reasoning` field is only present in gpt-oss via Ollama and OpenRouter.
        # - https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot#chat-completions-api
        # - https://openrouter.ai/docs/use-cases/reasoning-tokens#basic-usage-with-reasoning-tokens
        if reasoning := getattr(choice.message, 'reasoning', None):
            items.append(ThinkingPart(id='reasoning', content=reasoning, provider_name=self.system))

        # NOTE: We don't currently handle OpenRouter `reasoning_details`:
        # - https://openrouter.ai/docs/use-cases/reasoning-tokens#preserving-reasoning-blocks
        # If you need this, please file an issue.

        if choice.message.content:
            items.extend(
                (replace(part, id='content', provider_name=self.system) if isinstance(part, ThinkingPart) else part)
                for part in split_content_into_text_and_thinking(choice.message.content, self.profile.thinking_tags)
            )
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                items.append(ToolCallPart(c.get("function").get("name"), c.get("function").get("arguments"), tool_call_id=c.get("id")))

        raw_finish_reason = choice.finish_reason
        finish_reason = _CHAT_FINISH_REASON_MAP.get(raw_finish_reason)

        return ModelResponse(
            parts=items,
            usage=_map_usage(response, self._provider, "", self._model_name),
            model_name=response.model,
            timestamp=timestamp,
            provider_details=None,
            provider_response_id=response.id,
            provider_name=self._provider,
            finish_reason=finish_reason,
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
    pydantic_ai.models.OpenAIChatModel = AIModel