from __future__ import annotations

import json

from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, FinishReason
from ...typing import Union, Optional, AsyncResult, Messages
from ...requests.raise_for_status import raise_for_status
from ...requests import StreamSession
from ...errors import MissingAuthError, ResponseError

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
        stop: Union[str, list[str]] = None,
        stream: bool = False,
        headers: dict = None,
        extra_data: dict = {},
        **kwargs
    ) -> AsyncResult:
        if cls.needs_auth and api_key is None:
            raise MissingAuthError('Add a "api_key"')
        async with StreamSession(
            proxies={"all": proxy},
            headers=cls.get_headers(stream, api_key, headers),
            timeout=timeout
        ) as session:
            data = filter_none(
                messages=messages,
                model=cls.get_model(model),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                stream=stream,
                **extra_data
            )
            async with session.post(f"{api_base.rstrip('/')}/chat/completions", json=data) as response:
                await raise_for_status(response)
                if not stream:
                    data = await response.json()
                    choice = data["choices"][0]
                    if "content" in choice["message"]:
                        yield choice["message"]["content"].strip()
                    finish = cls.read_finish_reason(choice)
                    if finish is not None:
                        yield finish
                else:
                    first = True
                    async for line in response.iter_lines():
                        if line.startswith(b"data: "):
                            chunk = line[6:]
                            if chunk == b"[DONE]":
                                break
                            data = json.loads(chunk)
                            if "error_message" in data:
                                raise ResponseError(data["error_message"])
                            choice = data["choices"][0]
                            if "content" in choice["delta"] and choice["delta"]["content"]:
                                delta = choice["delta"]["content"]
                                if first:
                                    delta = delta.lstrip()
                                if delta:
                                    first = False
                                    yield delta
                            finish = cls.read_finish_reason(choice)
                            if finish is not None:
                                yield finish

    @staticmethod
    def read_finish_reason(choice: dict) -> Optional[FinishReason]:
        if "finish_reason" in choice and choice["finish_reason"] is not None:
            return FinishReason(choice["finish_reason"])

    @classmethod
    def get_headers(cls, stream: bool, api_key: str = None, headers: dict = None) -> dict:
        return {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            **(
                {"Authorization": f"Bearer {api_key}"}
                if cls.needs_auth and api_key is not None
                else {}
            ),
            **({} if headers is None else headers)
        }

def filter_none(**kwargs) -> dict:
    return {
        key: value
        for key, value in kwargs.items()
        if value is not None
    }