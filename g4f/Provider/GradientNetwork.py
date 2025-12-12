from __future__ import annotations

import json

from ..typing import AsyncResult, Messages
from ..providers.response import Reasoning, JsonResponse
from ..requests import StreamSession
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin


class GradientNetwork(AsyncGeneratorProvider, ProviderModelMixin):
    """
    Provider for chat.gradient.network
    Supports streaming text generation with Qwen and GPT OSS models.
    """
    label = "Gradient Network"
    url = "https://chat.gradient.network"
    api_endpoint = "https://chat.gradient.network/api/generate"

    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = "GPT OSS 120B"
    models = [
        default_model,
        "Qwen3 235B",
    ]
    model_aliases = {
        "qwen-3-235b": "Qwen3 235B",
        "qwen3-235b": "Qwen3 235B",
        "gpt-oss-120b": "GPT OSS 120B",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        enable_thinking: bool = True,
        **kwargs
    ) -> AsyncResult:
        """
        Create an async generator for streaming chat responses.

        Args:
            model: The model name to use
            messages: List of message dictionaries
            proxy: Optional proxy URL
            enable_thinking: Enable the thinking/analysis channel (maps to enableThinking in API)
            **kwargs: Additional arguments

        Yields:
            str: Content chunks from the response
            Reasoning: Reasoning content when enable_thinking is True
        """
        model = cls.get_model(model)

        headers = {
            "Accept": "application/x-ndjson",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Referer": f"{cls.url}/",
        }

        payload = {
            "clusterMode": "nvidia" if "GPT OSS" in model else "hybrid",
            "model": model,
            "messages": messages,
        }
        if enable_thinking:
            payload["enableThinking"] = enable_thinking
        async with StreamSession(headers=headers, proxy=proxy, impersonate="chrome") as session:
            async with session.post(
                cls.api_endpoint,
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        yield JsonResponse.from_dict(data)
                        msg_type = data.get("type")

                        if msg_type == "reply":
                            # Response chunks with content or reasoningContent
                            reply_data = data.get("data", {})
                            content = reply_data.get("content")
                            reasoning_content = reply_data.get("reasoningContent")

                            if reasoning_content:
                                yield Reasoning(reasoning_content)
                            if content:
                                yield content

                        # Skip clusterInfo and blockUpdate GPU visualization messages

                    except json.JSONDecodeError:
                        # Skip non-JSON lines (may be partial data or empty)
                        raise
