from __future__ import annotations

import hashlib
import time
import uuid
import json
from datetime import datetime
from aiohttp import ClientSession

from ..typing import SHA256, AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider


class Ails(AsyncGeneratorProvider):
    url: str              = "https://ai.ls"
    working               = True
    supports_gpt_35_turbo = True

    @staticmethod
    async def create_async_generator(
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "authority": "api.caipacity.com",
            "accept": "*/*",
            "accept-language": "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "authorization": "Bearer free",
            "client-id": str(uuid.uuid4()),
            "client-v": "0.1.278",
            "content-type": "application/json",
            "origin": "https://ai.ls",
            "referer": "https://ai.ls/",
            "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "cross-site",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "from-url": "https://ai.ls/?chat=1"
        }
        async with ClientSession(
            headers=headers
        ) as session:
            timestamp = _format_timestamp(int(time.time() * 1000))
            json_data = {
                "model": "gpt-3.5-turbo",
                "temperature": kwargs.get("temperature", 0.6),
                "stream": True,
                "messages": messages,
                "d": datetime.now().strftime("%Y-%m-%d"),
                "t": timestamp,
                "s": _hash({"t": timestamp, "m": messages[-1]["content"]}),
            }
            async with session.post(
                "https://api.caipacity.com/v1/chat/completions",
                proxy=proxy,
                json=json_data
            ) as response:
                response.raise_for_status()
                start = "data: "
                async for line in response.content:
                    line = line.decode('utf-8')
                    if line.startswith(start) and line != "data: [DONE]":
                        line = line[len(start):-1]
                        line = json.loads(line)
                        token = line["choices"][0]["delta"].get("content")
                        if token:
                            if "ai.ls" in token or "ai.ci" in token:
                                raise Exception("Response Error: " + token)
                            yield token


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("temperature", "float"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"


def _hash(json_data: dict[str, str]) -> SHA256:
    base_string: str = "%s:%s:%s:%s" % (
        json_data["t"],
        json_data["m"],
        "WI,2rU#_r:r~aF4aJ36[.Z(/8Rv93Rf",
        len(json_data["m"]),
    )

    return SHA256(hashlib.sha256(base_string.encode()).hexdigest())


def _format_timestamp(timestamp: int) -> str:
    e = timestamp
    n = e % 10
    r = n + 1 if n % 2 == 0 else n
    return str(e - n + r)