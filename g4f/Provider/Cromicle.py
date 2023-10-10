from __future__ import annotations

from aiohttp import ClientSession
from hashlib import sha256
from ..typing import AsyncResult, Messages, Dict

from .base_provider import AsyncGeneratorProvider
from .helper import format_prompt


class Cromicle(AsyncGeneratorProvider):
    url: str = 'https://cromicle.top'
    working: bool = True
    supports_gpt_35_turbo: bool = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        async with ClientSession(
            headers=_create_header()
        ) as session:
            async with session.post(
                f'{cls.url}/chat',
                proxy=proxy,
                json=_create_payload(format_prompt(messages))
            ) as response:
                response.raise_for_status()
                async for stream in response.content.iter_any():
                    if stream:
                        yield stream.decode()


def _create_header() -> Dict[str, str]:
    return {
        'accept': '*/*',
        'content-type': 'application/json',
    }


def _create_payload(message: str) -> Dict[str, str]:
    return {
        'message': message,
        'token': 'abc',
        'hash': sha256('abc'.encode() + message.encode()).hexdigest()
    }