from __future__ import annotations

import json
from typing import AsyncGenerator, Optional, List, Dict, Union, Any
from aiohttp import ClientSession, BaseConnector, ClientResponse

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_random_string, get_connector
from ..requests import raise_for_status

class Koala(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://koala.sh"
    working = True
    supports_message_history = True
    supports_gpt_4 = True
    default_model = 'gpt-4o-mini'

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: Optional[str] = None,
        connector: Optional[BaseConnector] = None,
        **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Union[str, int, float, List[Dict[str, Any]], None]], None]:
        if not model:
            model = "gpt-3.5-turbo"

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
            "Accept": "text/event-stream",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"{cls.url}/chat",
            "Flag-Real-Time-Data": "false",
            "Visitor-ID": get_random_string(20),
            "Origin": cls.url,
            "Alt-Used": "koala.sh",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "TE": "trailers",
        }

        async with ClientSession(headers=headers, connector=get_connector(connector, proxy)) as session:
            input_text = messages[-1]["content"]
            system_messages = " ".join(
                message["content"] for message in messages if message["role"] == "system"
            )
            if system_messages:
                input_text += f" {system_messages}"

            data = {
                "input": input_text,
                "inputHistory": [
                    message["content"]
                    for message in messages[:-1]
                    if message["role"] == "user"
                ],
                "outputHistory": [
                    message["content"]
                    for message in messages
                    if message["role"] == "assistant"
                ],
                "model": model,
            }

            async with session.post(f"{cls.url}/api/gpt/", json=data, proxy=proxy) as response:
                await raise_for_status(response)
                async for chunk in cls._parse_event_stream(response):
                    yield chunk

    @staticmethod
    async def _parse_event_stream(response: ClientResponse) -> AsyncGenerator[Dict[str, Any], None]:
        async for chunk in response.content:
            if chunk.startswith(b"data: "):
                yield json.loads(chunk[6:])
