from __future__ import annotations

from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, filter_none
from ...typing import AsyncResult, Messages
from ...requests import raise_for_status
from ...requests.aiohttp import StreamSession
from ...errors import ResponseError, MissingAuthError

class Replicate(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://replicate.com"
    working = True
    needs_auth = True
    default_model = "meta/meta-llama-3-70b-instruct"
    model_aliases = {
        "meta-llama/Meta-Llama-3-70B-Instruct": default_model
    }

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
        headers: dict = {
            "accept": "application/json",
        },
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        if cls.needs_auth and api_key is None:
            raise MissingAuthError("api_key is missing")
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
            api_base = "https://api.replicate.com/v1/models/"
        else:
            api_base = "https://replicate.com/api/models/"
        async with StreamSession(
            proxy=proxy,
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
            url = f"{api_base.rstrip('/')}/{model}/predictions"
            async with session.post(url, json=data) as response:
                message = "Model not found" if response.status == 404 else None
                await raise_for_status(response, message)
                result = await response.json()
                if "id" not in result:
                    raise ResponseError(f"Invalid response: {result}")
                async with session.get(result["urls"]["stream"], headers={"Accept": "text/event-stream"}) as response:
                    await raise_for_status(response)
                    event = None
                    async for line in response.iter_lines():
                        if line.startswith(b"event: "):
                            event = line[7:]
                            if event == b"done":
                                break
                        elif event == b"output":
                            if line.startswith(b"data: "):
                                new_text = line[6:].decode()
                                if new_text:
                                    yield new_text
                                else:
                                    yield "\n"
