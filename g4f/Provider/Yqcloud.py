from __future__ import annotations

from aiohttp import ClientSession

from ..typing import AsyncGenerator
from .base_provider import AsyncGeneratorProvider, format_prompt


class Yqcloud(AsyncGeneratorProvider):
    url = "https://chat9.yqcloud.top/"
    working = True
    supports_gpt_35_turbo = True

    @staticmethod
    async def create_async_generator(
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        timeout: int = 30,
        **kwargs,
    ) -> AsyncGenerator:
        async with ClientSession(
            headers=_create_header(), timeout=timeout
        ) as session:
            payload = _create_payload(messages)
            async with session.post("https://api.aichatos.cloud/api/generateStream", proxy=proxy, json=payload) as response:
                response.raise_for_status()
                async for stream in response.content.iter_any():
                    if stream:
                        yield stream.decode()


def _create_header():
    return {
        "accept"        : "application/json, text/plain, */*",
        "content-type"  : "application/json",
        "origin"        : "https://chat9.yqcloud.top",
    }


def _create_payload(messages: list[dict[str, str]]):
    return {
        "prompt": format_prompt(messages),
        "network": True,
        "system": "",
        "withoutContext": False,
        "stream": True,
        "userId": "#/chat/1693025544336"
    }
