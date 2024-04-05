from __future__ import annotations

import json

from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, FinishReason
from ...typing import AsyncResult, Messages
from ...requests.raise_for_status import raise_for_status
from ...requests import StreamSession
from ...errors import MissingAuthError

class Openai(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://openai.com"
    working = True
    needs_auth = True
    supports_message_history = True
    supports_system_message = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        api_key: str = None,
        api_base: str = "https://api.openai.com/v1",
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        stop: str = None,
        stream: bool = False,
        **kwargs
    ) -> AsyncResult:
        if api_key is None:
            raise MissingAuthError('Add a "api_key"')
        async with StreamSession(
            proxies={"all": proxy},
            headers=cls.get_headers(api_key),
            timeout=timeout
        ) as session:
            data = {
                "messages": messages,
                "model": cls.get_model(model),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stop": stop,
                "stream": stream,
            }
            async with session.post(f"{api_base.rstrip('/')}/chat/completions", json=data) as response:
                await raise_for_status(response)
                async for line in response.iter_lines():
                    if line.startswith(b"data: ") or not stream:
                        async for chunk in cls.read_line(line[6:] if stream else line, stream):
                            yield chunk

    @staticmethod
    async def read_line(line: str, stream: bool):
        if line == b"[DONE]":
            return
        choice = json.loads(line)["choices"][0]
        if stream and "content" in choice["delta"] and choice["delta"]["content"]:
            yield choice["delta"]["content"]
        elif not stream and "content" in choice["message"]:
            yield choice["message"]["content"]
        if "finish_reason" in choice and choice["finish_reason"] is not None:
            yield FinishReason(choice["finish_reason"])

    @staticmethod
    def get_headers(api_key: str) -> dict:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }