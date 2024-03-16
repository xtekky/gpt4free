from __future__ import annotations

import json
from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_connector
from ..errors import RateLimitError, ModelNotFoundError
from ..requests.raise_for_status import raise_for_status

class HuggingFace(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://huggingface.co/chat"
    working = True
    supports_message_history = True
    default_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        proxy: str = None,
        connector: BaseConnector = None,
        api_base: str = "https://api-inference.huggingface.co",
        api_key: str = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        headers = {}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        params = {
            "return_full_text": False,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            **kwargs
        }
        payload = {"inputs": format_prompt(messages), "parameters": params, "stream": stream}
        async with ClientSession(
            headers=headers,
            connector=get_connector(connector, proxy)
        ) as session:
            async with session.post(f"{api_base.rstrip('/')}/models/{model}", json=payload) as response:
                if response.status == 404:
                    raise ModelNotFoundError(f"Model is not supported: {model}")
                await raise_for_status(response)
                if stream:
                    first = True
                    async for line in response.content:
                        if line.startswith(b"data:"):
                            data = json.loads(line[5:])
                            if not data["token"]["special"]:
                                chunk = data["token"]["text"]
                                if first:
                                    first = False
                                    chunk = chunk.lstrip()
                                yield chunk
                else:
                    yield (await response.json())[0]["generated_text"].strip()

def format_prompt(messages: Messages) -> str:
    system_messages = [message["content"] for message in messages if message["role"] == "system"]
    question = " ".join([messages[-1]["content"], *system_messages])
    history = "".join([
        f"<s>[INST]{messages[idx-1]['content']} [/INST] {message['content']}</s>"
        for idx, message in enumerate(messages)
        if message["role"] == "assistant"
    ])
    return f"{history}<s>[INST] {question} [/INST]"