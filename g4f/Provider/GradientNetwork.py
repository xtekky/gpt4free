from __future__ import annotations

import json

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from ..providers.response import Reasoning
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin


class GradientNetwork(AsyncGeneratorProvider, ProviderModelMixin):
    """
    Provider for chat.gradient.network
    Supports streaming text generation with various Qwen models.
    """
    label = "Gradient Network"
    url = "https://chat.gradient.network"
    api_endpoint = "https://chat.gradient.network/api/generate"

    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = "qwen3-235b"
    models = [
        default_model,
        "qwen3-32b",
        "deepseek-r1-0528",
        "deepseek-v3-0324",
        "llama-4-maverick",
    ]
    model_aliases = {
        "qwen-3-235b": "qwen3-235b",
        "deepseek-r1": "deepseek-r1-0528",
        "deepseek-v3": "deepseek-v3-0324",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        temperature: float = None,
        max_tokens: int = None,
        enable_thinking: bool = False,
        **kwargs
    ) -> AsyncResult:
        """
        Create an async generator for streaming chat responses.

        Args:
            model: The model name to use
            messages: List of message dictionaries
            proxy: Optional proxy URL
            temperature: Optional temperature parameter
            max_tokens: Optional max tokens parameter
            enable_thinking: Enable the thinking/analysis channel
            **kwargs: Additional arguments

        Yields:
            str: Content chunks from the response
            Reasoning: Thinking content when enable_thinking is True
        """
        model = cls.get_model(model)

        headers = {
            "Accept": "application/x-ndjson",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Origin": cls.url,
            "Referer": f"{cls.url}/",
        }

        payload = {
            "model": model,
            "messages": messages,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if enable_thinking:
            payload["enableThinking"] = True

        async with ClientSession(headers=headers) as session:
            async with session.post(
                cls.api_endpoint,
                json=payload,
                proxy=proxy
            ) as response:
                response.raise_for_status()

                async for line_bytes in response.content:
                    if not line_bytes:
                        continue

                    line = line_bytes.decode("utf-8").strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        msg_type = data.get("type")

                        if msg_type == "text":
                            # Regular text content
                            content = data.get("data")
                            if content:
                                yield content

                        elif msg_type == "thinking":
                            # Thinking/reasoning content
                            content = data.get("data")
                            if content:
                                yield Reasoning(content)

                        elif msg_type == "done":
                            # Stream complete
                            break

                        # Ignore clusterInfo and blockUpdate messages
                        # as they are for GPU cluster visualization only

                    except json.JSONDecodeError:
                        # Skip non-JSON lines
                        continue
