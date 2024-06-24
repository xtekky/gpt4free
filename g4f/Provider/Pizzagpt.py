import json
from aiohttp import ClientSession

from ..typing import Messages, AsyncResult
from .base_provider import AsyncGeneratorProvider

class Pizzagpt(AsyncGeneratorProvider):
    url = "https://www.pizzagpt.it"
    api_endpoint = "/api/chatx-completion"
    supports_message_history = False
    supports_gpt_35_turbo = True
    working = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        payload = {
            "question": messages[-1]["content"]
        }
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/json",
            "Origin": cls.url,
            "Referer": f"{cls.url}/en",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "X-Secret": "Marinara"
        }

        async with ClientSession() as session:
            async with session.post(
                f"{cls.url}{cls.api_endpoint}",
                json=payload,
                proxy=proxy,
                headers=headers
            ) as response:
                response.raise_for_status()
                response_json = await response.json()
                yield response_json["answer"]["content"]
