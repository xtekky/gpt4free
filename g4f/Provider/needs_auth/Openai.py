from __future__ import annotations

import json

from ..helper import filter_none
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin, FinishReason
from ...typing import Union, Optional, AsyncResult, Messages, ImageType
from ...requests import StreamSession, raise_for_status
from ...errors import MissingAuthError, ResponseError
from ...image import to_data_uri

class Openai(AsyncGeneratorProvider, ProviderModelMixin):
    label = "OpenAI API"
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
        image: ImageType = None,
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
        if image is not None:
            if not model and hasattr(cls, "default_vision_model"):
                model = cls.default_vision_model
            messages[-1]["content"] = [
                {
                    "type": "image_url",
                    "image_url": {"url": to_data_uri(image)}
                },
                {
                    "type": "text",
                    "text": messages[-1]["content"]
                }
            ]
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
                    cls.raise_error(data)
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
                            cls.raise_error(data)
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

    @staticmethod
    def raise_error(data: dict):
        if "error_message" in data:
            raise ResponseError(data["error_message"])
        elif "error" in data:
            raise ResponseError(f'Error {data["error"]["code"]}: {data["error"]["message"]}')

    @classmethod
    def get_headers(cls, stream: bool, api_key: str = None, headers: dict = None) -> dict:
        return {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            **(
                {"Authorization": f"Bearer {api_key}"}
                if api_key is not None else {}
            ),
            **({} if headers is None else headers)
        }