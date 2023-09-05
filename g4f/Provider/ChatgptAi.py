from __future__ import annotations

import re
import html
import json
from aiohttp import ClientSession

from ..typing       import AsyncGenerator
from .base_provider import AsyncGeneratorProvider


class ChatgptAi(AsyncGeneratorProvider):
    url: str = "https://chatgpt.ai/"
    working = True
    supports_gpt_35_turbo  = True
    _system_data = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        **kwargs
    ) -> AsyncGenerator:
        headers = {
            "authority"          : "chatgpt.ai",
            "accept"             : "*/*",
            "accept-language"    : "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "cache-control"      : "no-cache",
            "origin"             : "https://chatgpt.ai",
            "pragma"             : "no-cache",
            "referer"            : cls.url,
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
            if not cls._system_data:
                async with session.get(cls.url, proxy=proxy) as response:
                    response.raise_for_status()
                    match = re.findall(r"data-system='([^']+)'", await response.text())
                    if not match:
                        raise RuntimeError("No system data")
                    cls._system_data = json.loads(html.unescape(match[0]))

            data = {
                "botId":      cls._system_data["botId"],
                "clientId":   "",
                "contextId":  cls._system_data["contextId"],
                "id":         cls._system_data["id"],
                "messages":   messages[:-1],
                "newMessage": messages[-1]["content"],
                "session":    cls._system_data["sessionId"],
                "stream":     True
            }
            async with session.post(
                "https://chatgpt.ai/wp-json/mwai-ui/v1/chats/submit",
                proxy=proxy,
                json=data
            ) as response:
                response.raise_for_status()
                start = "data: "
                async for line in response.content:
                    line = line.decode('utf-8')
                    if line.startswith(start):
                        line = json.loads(line[len(start):-1])
                        if line["type"] == "live":
                            yield line["data"]