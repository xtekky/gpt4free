from __future__ import annotations

import json
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider
from ..helper import format_prompt


class Chatgpt4Online(AsyncGeneratorProvider):
    url = "https://chatgpt4online.org"
    api_endpoint = "/wp-json/mwai-ui/v1/chats/submit"
    working = False
    
    default_model = 'gpt-4'
    models = [default_model]
    
    async def get_nonce(headers: dict) -> str:
        async with ClientSession(headers=headers) as session:
            async with session.post(f"https://chatgpt4online.org/wp-json/mwai/v1/start_session") as response:
                return (await response.json())["restNonce"]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "accept": "text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "dnt": "1",
            "origin": cls.url,
            "priority": "u=1, i",
            "referer": f"{cls.url}/",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        }
        headers['x-wp-nonce'] = await cls.get_nonce(headers)
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "botId": "default",
                "newMessage": prompt,
                "stream": True,
            }

            async with session.post(f"{cls.url}{cls.api_endpoint}", json=data, proxy=proxy) as response:
                response.raise_for_status()
                full_response = ""

                async for chunk in response.content.iter_any():
                    if chunk:
                        try:
                            # Extract the JSON object from the chunk
                            for line in chunk.decode().splitlines():
                                if line.startswith("data: "):
                                    json_data = json.loads(line[6:])
                                    if json_data["type"] == "live":
                                        full_response += json_data["data"]
                                    elif json_data["type"] == "end":
                                        final_data = json.loads(json_data["data"])
                                        full_response = final_data["reply"]
                                        break
                        except json.JSONDecodeError:
                            continue

                yield full_response

