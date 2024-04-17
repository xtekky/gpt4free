from __future__ import annotations

import re
import json
from aiohttp import ClientSession

from ..typing import Messages, AsyncResult
from ..requests import get_args_from_browser
from ..webdriver import WebDriver
from .base_provider import AsyncGeneratorProvider
from .helper import get_random_string

class Chatgpt4Online(AsyncGeneratorProvider):
    url = "https://chatgpt4online.org"
    supports_message_history = True
    supports_gpt_35_turbo = True
    working = True 
    _wpnonce = None
    _context_id = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        webdriver: WebDriver = None,
        **kwargs
    ) -> AsyncResult:
        args = get_args_from_browser(f"{cls.url}/chat/", webdriver, proxy=proxy)
        async with ClientSession(**args) as session:
            if not cls._wpnonce:
                async with session.get(f"{cls.url}/chat/", proxy=proxy) as response:
                    response.raise_for_status()
                    response = await response.text()
                    result = re.search(r'restNonce&quot;:&quot;(.*?)&quot;', response)
                    if result:
                        cls._wpnonce = result.group(1)
                    else:
                        raise RuntimeError("No nonce found")
                    result = re.search(r'contextId&quot;:(.*?),', response)
                    if result:
                        cls._context_id = result.group(1)
                    else:
                        raise RuntimeError("No contextId found")
            data = {
                "botId":"default",
                "customId":None,
                "session":"N/A",
                "chatId":get_random_string(11),
                "contextId":cls._context_id,
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
