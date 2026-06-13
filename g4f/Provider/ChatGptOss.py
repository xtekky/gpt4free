from __future__ import annotations

import json
import hashlib
import time
import random

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin


class ChatGptOss(AsyncGeneratorProvider, ProviderModelMixin):
    """
    Free provider for chat-gpt-oss.com
    Supports GPT-5-Nano and GPT-OSS-120B models via SSE streaming.
    """
    label = "ChatGptOss"
    url = "https://chat-gpt-oss.com"
    api_endpoint = "https://chat-gpt-oss.com/api/message"
    working = True
    needs_auth = False
    supports_stream = True
    supports_system_message = False
    supports_message_history = False

    default_model = "gpt-oss-120b"
    models = [
        "gpt-oss-120b",
        "gpt-5-nano",
    ]
    model_aliases = {
        "gpt-oss": "gpt-oss-120b",
    }

    @classmethod
    def _generate_fingerprint(cls) -> str:
        """Generate a random fingerprint hash similar to x-fingerprint header."""
        seed = f"{time.time()}-{random.random()}"
        return hashlib.md5(seed.encode()).hexdigest()

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)

        # Combine all messages into a single prompt
        # The API accepts a single "content" string, not a message history
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "\n".join(text_parts)
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(content)

        # Use only the last user message if there's no history context
        if len(messages) == 1:
            prompt = prompt_parts[0]
        else:
            prompt = "\n\n".join(prompt_parts)

        headers = {
            "accept": "text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": "https://chat-gpt-oss.com",
            "pragma": "no-cache",
            "referer": "https://chat-gpt-oss.com/",
            "sec-ch-ua": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
            "x-fingerprint": cls._generate_fingerprint(),
        }

        payload = {
            "conversation_id": None,
            "model": model,
            "content": prompt,
            "reasoning_effort": "low",
        }

        async with ClientSession(headers=headers) as session:
            async with session.post(
                cls.api_endpoint,
                json=payload,
                proxy=proxy,
            ) as response:
                response.raise_for_status()

                event_type = None
                async for line_bytes in response.content:
                    line = line_bytes.decode("utf-8").rstrip("\n").rstrip("\r")

                    if not line:
                        event_type = None
                        continue

                    if line.startswith("event:"):
                        event_type = line[len("event:"):].strip()
                        continue

                    if line.startswith("data:") and event_type == "message":
                        data_str = line[len("data:"):].strip()
                        if not data_str:
                            continue
                        try:
                            chunk = json.loads(data_str)
                            content = chunk.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

                    # Stop on summary event (end of response)
                    if event_type == "summary":
                        break
