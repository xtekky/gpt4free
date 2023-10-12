from __future__ import annotations

from aiohttp import ClientSession
import json

from ..typing       import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, format_prompt


class GptGo(AsyncGeneratorProvider):
    url                   = "https://gptgo.ai"
    supports_gpt_35_turbo = True
    working               = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "User-Agent"         : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            "Accept"             : "*/*",
            "Accept-language"    : "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "Origin"             : cls.url,
            "Referer"            : cls.url + "/",
            "Sec-Fetch-Dest"     : "empty",
            "Sec-Fetch-Mode"     : "cors",
            "Sec-Fetch-Site"     : "same-origin",
        }
        async with ClientSession(
            headers=headers
        ) as session:
            async with session.get(
                "https://gptgo.ai/action_get_token.php",
                params={
                    "q": format_prompt(messages),
                    "hlgpt": "default",
                    "hl": "en"
                },
                proxy=proxy
            ) as response:
                response.raise_for_status()
                token = (await response.json(content_type=None))["token"]

            async with session.get(
                "https://gptgo.ai/action_ai_gpt.php",
                params={
                    "token": token,
                },
                proxy=proxy
            ) as response:
                response.raise_for_status()
                start = "data: "
                async for line in response.content:
                    line = line.decode()
                    if line.startswith("data: "):
                        if line.startswith("data: [DONE]"):
                            break
                        line = json.loads(line[len(start):-1])
                        if line["choices"][0]["finish_reason"] == "stop":
                            break
                        content = line["choices"][0]["delta"].get("content")
                        if content:
                            yield content


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("proxy", "str"),
            ("temperature", "float"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"