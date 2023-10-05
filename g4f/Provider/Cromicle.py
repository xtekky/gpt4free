from __future__ import annotations

from aiohttp import ClientSession
from hashlib import sha256

from ..typing import AsyncGenerator
from .base_provider import AsyncGeneratorProvider


class Cromicle(AsyncGeneratorProvider):
    url                   = 'https://cromicle.top'
    working               = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        **kwargs
    ) -> AsyncGenerator:
        message = messages[-1]["content"]
        async with ClientSession(
            headers=_create_header()
        ) as session:
            async with session.post(
                cls.url + '/chat',
                proxy=proxy,
                json=_create_payload(message, **kwargs)
            ) as response:
                response.raise_for_status()
                async for stream in response.content.iter_any():
                    if stream:
                        yield stream.decode()


def _create_header():
    return {
        'accept': '*/*',
        'content-type': 'application/json',
    }


def _create_payload(message: str):
    return {
        'message'    : message,
        'token' : 'abc',
        'hash'    : sha256('abc'.encode() + message.encode()).hexdigest()
    }
