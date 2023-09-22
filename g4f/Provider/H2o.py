from __future__ import annotations

import json
import uuid

from aiohttp import ClientSession

from ..typing import AsyncGenerator
from .base_provider import AsyncGeneratorProvider, format_prompt


class H2o(AsyncGeneratorProvider):
    url = "https://gpt-gm.h2o.ai"
    working = True
    model = "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v1"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        **kwargs
    ) -> AsyncGenerator:
        model = model if model else cls.model
        headers = {"Referer": cls.url + "/"}

        async with ClientSession(
            headers=headers
        ) as session:
            data = {
                "ethicsModalAccepted": "true",
                "shareConversationsWithModelAuthors": "true",
                "ethicsModalAcceptedAt": "",
                "activeModel": model,
                "searchEnabled": "true",
            }
            async with session.post(
                f"{cls.url}/settings",
                proxy=proxy,
                data=data
            ) as response:
                response.raise_for_status()

            async with session.post(
                f"{cls.url}/conversation",
                proxy=proxy,
                json={"model": model},
            ) as response:
                response.raise_for_status()
                conversationId = (await response.json())["conversationId"]

            data = {
                "inputs": format_prompt(messages),
                "parameters": {
                    "temperature": 0.4,
                    "truncate": 2048,
                    "max_new_tokens": 1024,
                    "do_sample":  True,
                    "repetition_penalty": 1.2,
                    "return_full_text": False,
                    **kwargs
                },
                "stream": True,
                "options": {
                    "id": str(uuid.uuid4()),
                    "response_id": str(uuid.uuid4()),
                    "is_retry": False,
                    "use_cache": False,
                    "web_search_id": "",
                },
            }
            async with session.post(
                f"{cls.url}/conversation/{conversationId}",
                proxy=proxy,
                json=data
             ) as response:
                start = "data:"
                async for line in response.content:
                    line = line.decode("utf-8")
                    if line and line.startswith(start):
                        line = json.loads(line[len(start):-1])
                        if not line["token"]["special"]:
                            yield line["token"]["text"]

            async with session.delete(
                f"{cls.url}/conversation/{conversationId}",
                proxy=proxy,
                json=data
            ) as response:
                response.raise_for_status()


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("temperature", "float"),
            ("truncate", "int"),
            ("max_new_tokens", "int"),
            ("do_sample", "bool"),
            ("repetition_penalty", "float"),
            ("return_full_text", "bool"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
