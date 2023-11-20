from __future__ import annotations

from aiohttp import ClientSession
import json

from ...typing       import AsyncGenerator
from ..base_provider import AsyncGeneratorProvider


class CodeLinkAva(AsyncGeneratorProvider):
    url                   = "https://ava-ai-ef611.web.app"
    supports_gpt_35_turbo = True
    working               = False

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: list[dict[str, str]],
        **kwargs
    ) -> AsyncGenerator:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-language": "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "Origin": cls.url,
            "Referer": f"{cls.url}/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }
        async with ClientSession(
                headers=headers
            ) as session:
            data = {
                "messages": messages,
                "temperature": 0.6,
                "stream": True,
                **kwargs
            }
            async with session.post("https://ava-alpha-api.codelink.io/api/chat", json=data) as response:
                response.raise_for_status()
                async for line in response.content:
                    line = line.decode()
                    if line.startswith("data: "):
                        if line.startswith("data: [DONE]"):
                            break
                        line = json.loads(line[6:-1])

                        content = line["choices"][0]["delta"].get("content")
                        if content:
                            yield content