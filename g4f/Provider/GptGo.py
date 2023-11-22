from __future__ import annotations

from aiohttp import ClientSession
import json
import base64

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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-language": "en-US",
            "Origin": cls.url,
            "Referer": f"{cls.url}/",
            "sec-ch-ua": '"Google Chrome";v="116", "Chromium";v="116", "Not?A_Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }
        async with ClientSession(
                headers=headers
            ) as session:
            async with session.post(
                "https://gptgo.ai/get_token.php",
                data={"ask": format_prompt(messages)},
                proxy=proxy
            ) as response:
                response.raise_for_status()
                token = await response.text();
                token = base64.b64decode(token[10:-20]).decode()

            async with session.get(
                "https://api.gptgo.ai/web.php",
                params={"array_chat": token},
                proxy=proxy
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line.startswith(b"data: [DONE]"):
                        break
                    if line.startswith(b"data: "):
                        line = json.loads(line[6:])
                        content = line["choices"][0]["delta"].get("content")
                        if content and content != "\n#GPTGO ":
                            yield content
