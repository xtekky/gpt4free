from __future__ import annotations

import re
import json
from aiohttp import ClientSession

from ..typing import Messages, AsyncResult
from .base_provider import AsyncGeneratorProvider
from .helper import get_random_string

class Chatgpt4Online(AsyncGeneratorProvider):
    url = "https://chatgpt4online.org"
    supports_message_history = True
    supports_gpt_35_turbo = True
    working = True
    _wpnonce = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "accept": "*/*",
            "accept-language": "en-US",
            "content-type": "application/json",
            "sec-ch-ua": "\"Not_A Brand\";v=\"8\", \"Chromium\";v=\"120\", \"Google Chrome\";v=\"120\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "referer": "https://chatgpt4online.org/",
            "referrer-policy": "strict-origin-when-cross-origin"
        }
        async with ClientSession(headers=headers) as session:
            if not cls._wpnonce:
                async with session.get(f"{cls.url}/", proxy=proxy) as response:
                    response.raise_for_status()
                    response = await response.text()
                    result = re.search(r'restNonce&quot;:&quot;(.*?)&quot;', response)
                    if result:
                        cls._wpnonce = result.group(1)
                    else:
                        raise RuntimeError("No nonce found")
            data = {
                "botId":"default",
                "customId":None,
                "session":"N/A",
                "chatId":get_random_string(11),
                "contextId":58,
                "messages":messages[:-1],
                "newMessage":messages[-1]["content"],
                "newImageId":None,
                "stream":True
            }
            async with session.post(
                f"{cls.url}/wp-json/mwai-ui/v1/chats/submit",
                json=data,
                proxy=proxy,
                headers={"x-wp-nonce": cls._wpnonce}
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line.startswith(b"data: "):
                        line = json.loads(line[6:])
                        if "type" not in line:
                            raise RuntimeError(f"Response: {line}")
                        elif line["type"] == "live":
                            yield line["data"]
                        elif line["type"] == "end":
                            break