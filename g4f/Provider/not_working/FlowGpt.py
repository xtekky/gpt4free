from __future__ import annotations

import json
import time
import hashlib
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import get_random_hex, get_random_string
from ...requests.raise_for_status import raise_for_status

class FlowGpt(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://flowgpt.com/chat"
    working = False
    supports_message_history = True
    supports_system_message = True
    default_model = "gpt-3.5-turbo"
    models = [
        "gpt-3.5-turbo",
        "gpt-3.5-long",
        "gpt-4-turbo",
        "google-gemini",
        "claude-instant",
        "claude-v1",
        "claude-v2",
        "llama2-13b",
        "mythalion-13b",
        "pygmalion-13b",
        "chronos-hermes-13b",
        "Mixtral-8x7B",
        "Dolphin-2.6-8x7B",
    ]
    model_aliases = {
        "gemini": "google-gemini",
        "gemini-pro": "google-gemini"
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        timestamp = str(int(time.time()))
        auth = "Bearer null"
        nonce = get_random_hex()
        data = f"{timestamp}-{nonce}-{auth}"
        signature = hashlib.md5(data.encode()).hexdigest()

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
            "Accept": "*/*",
            "Accept-Language": "en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://flowgpt.com/",
            "Content-Type": "application/json",
            "Authorization": "Bearer null",
            "Origin": "https://flowgpt.com",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "TE": "trailers",
            "Authorization": auth,
            "x-flow-device-id": f"f-{get_random_string(19)}",
            "x-nonce": nonce,
            "x-signature": signature,
            "x-timestamp": timestamp
        }
        async with ClientSession(headers=headers) as session:
            history = [message for message in messages[:-1] if message["role"] != "system"]
            system_message = "\n".join([message["content"] for message in messages if message["role"] == "system"])
            if not system_message:
                system_message = "You are helpful assistant. Follow the user's instructions carefully."
            data = {
                "model": model,
                "nsfw": False,
                "question": messages[-1]["content"],
                "history": [{"role": "assistant", "content": "Hello, how can I help you today?"}, *history],
                "system": system_message,
                "temperature": temperature,
                "promptId": f"model-{model}",
                "documentIds": [],
                "chatFileDocumentIds": [],
                "generateImage": False,
                "generateAudio": False
            }
            async with session.post("https://prod-backend-k8s.flowgpt.com/v3/chat-anonymous", json=data, proxy=proxy) as response:
                await raise_for_status(response)
                async for chunk in response.content:
                    if chunk.strip():
                        message = json.loads(chunk)
                        if "event" not in message:
                            continue
                        if message["event"] == "text":
                            yield message["data"]
