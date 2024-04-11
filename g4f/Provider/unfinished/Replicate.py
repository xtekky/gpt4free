from __future__ import annotations

import asyncio

from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, filter_none
from ...typing import AsyncResult, Messages
from ...requests import StreamSession, raise_for_status
from ...image import ImageResponse
from ...errors import ResponseError, MissingAuthError

class Replicate(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://replicate.com"
    working = True
    default_model = "mistralai/mixtral-8x7b-instruct-v0.1"
    api_base = "https://api.replicate.com/v1/models/"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        api_key: str = None,
        proxy: str = None,
        timeout: int = 180,
        system_prompt: str = None,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: float = None,
        stop: list = None,
        extra_data: dict = {},
        headers: dict = {},
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        if api_key is None:
            raise MissingAuthError("api_key is missing")
        headers["Authorization"] = f"Bearer {api_key}"
        async with StreamSession(
            proxies={"all": proxy},
            headers=headers,
            timeout=timeout
        ) as session:
            data = {
                "stream": True,
                "input": {
                    "prompt": format_prompt(messages),
                    **filter_none(
                        system_prompt=system_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stop_sequences=",".join(stop) if stop else None
                    ),
                    **extra_data
                },
            }
            url = f"{cls.api_base.rstrip('/')}/{model}/predictions"
            async with session.post(url, json=data) as response:
                await raise_for_status(response)
                result = await response.json()
            if "id" not in result:
                raise ResponseError(f"Invalid response: {result}")
            async with session.get(result["urls"]["stream"], headers={"Accept": "text/event-stream"}) as response:
                await raise_for_status(response)
                event = None
                async for line in response.iter_lines():
                    if line.startswith(b"event: "):
                        event = line[7:]
                    elif event == b"output":
                        if line.startswith(b"data: "):
                            yield line[6:].decode()
                        elif not line.startswith(b"id: "):
                            continue#yield "+"+line.decode()
                    elif event == b"done":
                        break