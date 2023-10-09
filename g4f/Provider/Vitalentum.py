from __future__ import annotations

import json
from aiohttp import ClientSession

from .base_provider import AsyncGeneratorProvider
from ..typing import AsyncResult, Messages

class Vitalentum(AsyncGeneratorProvider):
    url                   = "https://app.vitalentum.io"
    working               = True
    supports_gpt_35_turbo = True


    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "User-Agent"         : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            "Accept"             : "text/event-stream",
            "Accept-language"    : "de,en-US;q=0.7,en;q=0.3",
            "Origin"             : cls.url,
            "Referer"            : cls.url + "/",
            "Sec-Fetch-Dest"     : "empty",
            "Sec-Fetch-Mode"     : "cors",
            "Sec-Fetch-Site"     : "same-origin",
        }
        conversation = json.dumps({"history": [{
            "speaker": "human" if message["role"] == "user" else "bot",
            "text": message["content"],
        } for message in messages]})
        data = {
            "conversation": conversation,
            "temperature": 0.7,
            **kwargs
        }
        async with ClientSession(
            headers=headers
        ) as session:
            async with session.post(f"{cls.url}/api/converse-edge", json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for line in response.content:
                    line = line.decode()
                    if line.startswith("data: "):
                        if line.startswith("data: [DONE]"):
                            break
                        line = json.loads(line[6:-1])
                        content = line["choices"][0]["delta"].get("content")
                        if content:
                            yield content


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("proxy", "str"),
            ("temperature", "float"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"