from __future__ import annotations

import random
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, format_prompt


class Yqcloud(AsyncGeneratorProvider):
    url = "https://chat9.yqcloud.top/"
    working = True
    supports_gpt_35_turbo = True

    @staticmethod
    async def create_async_generator(
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs,
    ) -> AsyncResult:
        async with ClientSession(
            headers=_create_header()
        ) as session:
            payload = _create_payload(messages, **kwargs)
            async with session.post("https://api.aichatos.cloud/api/generateStream", proxy=proxy, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_any():
                    if chunk:
                        chunk = chunk.decode()
                        if "sorry, 您的ip已由于触发防滥用检测而被封禁" in chunk:
                            raise RuntimeError("IP address is blocked by abuse detection.")
                        yield chunk


def _create_header():
    return {
        "accept"        : "application/json, text/plain, */*",
        "content-type"  : "application/json",
        "origin"        : "https://chat9.yqcloud.top",
    }


def _create_payload(
    messages: Messages,
    system_message: str = "",
    user_id: int = None,
    **kwargs
):
    if not user_id:
        user_id = random.randint(1690000544336, 2093025544336)
    return {
        "prompt": format_prompt(messages),
        "network": True,
        "system": system_message,
        "withoutContext": False,
        "stream": True,
        "userId": f"#/chat/{user_id}"
    }
