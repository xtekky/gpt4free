from __future__ import annotations

import random
from ..requests import StreamSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, format_prompt


class Binjie(AsyncGeneratorProvider):
    url = "https://chat18.aichatos8.com"
    working = True
    supports_gpt_4 = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    @staticmethod
    async def create_async_generator(
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        **kwargs,
    ) -> AsyncResult:
        async with StreamSession(
            headers=_create_header(), proxies={"https": proxy}, timeout=timeout
        ) as session:
            payload = _create_payload(messages, **kwargs)
            async with session.post("https://api.binjie.fun/api/generateStream", json=payload) as response:
                response.raise_for_status()
                async for chunk in response.iter_content():
                    if chunk:
                        chunk = chunk.decode()
                        if "sorry, 您的ip已由于触发防滥用检测而被封禁" in chunk:
                            raise RuntimeError("IP address is blocked by abuse detection.")
                        yield chunk


def _create_header():
    return {
        "accept"        : "application/json, text/plain, */*",
        "content-type"  : "application/json",
        "origin"        : "https://chat18.aichatos8.com",
        "referer"       : "https://chat18.aichatos8.com/"
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

