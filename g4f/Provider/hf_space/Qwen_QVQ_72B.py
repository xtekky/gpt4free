from __future__ import annotations

import json
from aiohttp import ClientSession, FormData

from ...typing import AsyncResult, Messages, ImagesType
from ...requests import raise_for_status
from ...errors import ResponseError
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, get_random_string
from ...image import to_bytes, is_accepted_format

class Qwen_QVQ_72B(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://qwen-qvq-72b-preview.hf.space"
    api_endpoint = "/gradio_api/call/generate"

    working = True

    default_model = "qwen-qvq-72b-preview"
    models = [default_model]
    model_aliases = {"qvq-72b": default_model}
    vision_models = models

    @classmethod
    async def create_async_generator(
        cls, model: str, messages: Messages,
        images: ImagesType = None,
        api_key: str = None, 
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "Accept": "application/json",
        }
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        async with ClientSession(headers=headers) as session:
            if images:
                data = FormData()
                data_bytes = to_bytes(images[0][0])
                data.add_field("files", data_bytes, content_type=is_accepted_format(data_bytes), filename=images[0][1])
                url = f"{cls.url}/gradio_api/upload?upload_id={get_random_string()}"
                async with session.post(url, data=data, proxy=proxy) as response:
                    await raise_for_status(response)
                    image = await response.json()
                data = {"data": [{"path": image[0]}, format_prompt(messages)]}
            else:
                data = {"data": [None, format_prompt(messages)]}
            async with session.post(f"{cls.url}{cls.api_endpoint}", json=data, proxy=proxy) as response:
                await raise_for_status(response)
                event_id = (await response.json()).get("event_id")
                async with session.get(f"{cls.url}{cls.api_endpoint}/{event_id}") as event_response:
                    await raise_for_status(event_response)
                    event = None
                    text_position = 0
                    async for chunk in event_response.content:
                        if chunk.startswith(b"event: "):
                            event = chunk[7:].decode(errors="replace").strip()
                        if chunk.startswith(b"data: "):
                            if event == "error":
                                raise ResponseError(f"GPU token limit exceeded: {chunk.decode(errors='replace')}")
                            if event in ("complete", "generating"):
                                try:
                                    data = json.loads(chunk[6:])
                                except (json.JSONDecodeError, KeyError, TypeError) as e:
                                    raise RuntimeError(f"Failed to read response: {chunk.decode(errors='replace')}", e)
                                if event == "generating":
                                    if isinstance(data[0], str):
                                        yield data[0][text_position:]
                                        text_position = len(data[0])
                                else:
                                    break
