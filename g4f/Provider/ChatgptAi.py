from __future__ import annotations

import re, html, json, string, random
from aiohttp import ClientSession

from ..typing import Messages, AsyncResult
from ..errors import RateLimitError
from .base_provider import AsyncGeneratorProvider
from .helper import get_random_string

class ChatgptAi(AsyncGeneratorProvider):
    url = "https://chatgpt.ai"
    working = True
    supports_message_history = True
    supports_system_message = True,
    supports_gpt_4 = True,
    _system = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "authority"          : "chatgpt.ai",
            "accept"             : "*/*",
            "accept-language"    : "en-US",
            "cache-control"      : "no-cache",
            "origin"             : cls.url,
            "pragma"             : "no-cache",
            "referer"            : f"{cls.url}/",
            "sec-ch-ua"          : '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
            "sec-ch-ua-mobile"   : "?0",
            "sec-ch-ua-platform" : '"Windows"',
            "sec-fetch-dest"     : "empty",
            "sec-fetch-mode"     : "cors",
            "sec-fetch-site"     : "same-origin",
            "user-agent"         : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        }
        async with ClientSession(
                headers=headers
            ) as session:
            if not cls._system:
                async with session.get(cls.url, proxy=proxy) as response:
                    response.raise_for_status()
                    text = await response.text()
                result = re.search(r"data-system='(.*?)'", text)
                if result :
                    cls._system = json.loads(html.unescape(result.group(1)))
            if not cls._system:
                raise RuntimeError("System args not found")

            data = {
                "botId": cls._system["botId"],
                "customId": cls._system["customId"],
                "session": cls._system["sessionId"],
                "chatId": get_random_string(),
                "contextId": cls._system["contextId"],
                "messages": messages[:-1],
                "newMessage": messages[-1]["content"],
                "newFileId": None,
                "stream":True
            }
            async with session.post(
               "https://chatgate.ai/wp-json/mwai-ui/v1/chats/submit",
                proxy=proxy,
                json=data,
                headers={"X-Wp-Nonce": cls._system["restNonce"]}
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line.startswith(b"data: "):
                        try:
                            line = json.loads(line[6:])
                            assert "type" in line
                        except:
                            raise RuntimeError(f"Broken line: {line.decode()}")
                        if line["type"] == "error":
                            if "https://chatgate.ai/login" in line["data"]:
                                raise RateLimitError("Rate limit reached")
                            raise RuntimeError(line["data"])
                        if line["type"] == "live":
                            yield line["data"]
                        elif line["type"] == "end":
                            break