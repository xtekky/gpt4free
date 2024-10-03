from __future__ import annotations

from aiohttp import ClientSession
import json

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from .helper import format_prompt


class Allyfy(AsyncGeneratorProvider):
    url = "https://allyfy.chat"
    api_endpoint = "https://chatbot.allyfy.chat/api/v1/message/stream/super/chat"
    working = True
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
            "accept": "text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json;charset=utf-8",
            "dnt": "1",
            "origin": "https://www.allyfy.chat",
            "priority": "u=1, i",
            "referer": "https://www.allyfy.chat/",
            "referrer": "https://www.allyfy.chat",
            'sec-ch-ua': '"Not/A)Brand";v="8", "Chromium";v="126"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        }
        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            data = {
                "messages": [{"content": prompt, "role": "user"}],
                "content": prompt,
                "baseInfo": {
                    "clientId": "q08kdrde1115003lyedfoir6af0yy531",
                    "pid": "38281",
                    "channelId": "100000",
                    "locale": "en-US",
                    "localZone": 180,
                    "packageName": "com.cch.allyfy.webh",
                }
            }
            async with session.post(f"{cls.api_endpoint}", json=data, proxy=proxy) as response:
                response.raise_for_status()
                full_response = []
                async for line in response.content:
                    line = line.decode().strip()
                    if line.startswith("data:"):
                        data_content = line[5:]
                        if data_content == "[DONE]":
                            break
                        try:
                            json_data = json.loads(data_content)
                            if "content" in json_data:
                                full_response.append(json_data["content"])
                        except json.JSONDecodeError:
                            continue
                yield "".join(full_response)
