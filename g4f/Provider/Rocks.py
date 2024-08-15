import json
from aiohttp import ClientSession

from ..typing import Messages, AsyncResult
from .base_provider import AsyncGeneratorProvider

class Rocks(AsyncGeneratorProvider):
    url = "https://api.discord.rocks"
    api_endpoint = "/chat/completions"
    supports_message_history = False
    supports_gpt_35_turbo = True
    working = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        payload = {"messages":messages,"model":model,"max_tokens":4096,"temperature":1,"top_p":1,"stream":True}

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": cls.url,
            "Referer": f"{cls.url}/en",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        }

        async with ClientSession() as session:
            async with session.post(
                f"{cls.url}{cls.api_endpoint}",
                json=payload,
                proxy=proxy,
                headers=headers
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line.startswith(b"data: "):
                        try:
                            line = json.loads(line[6:])
                        except:
                            continue
                        chunk = line["choices"][0]["delta"].get("content")
                        if chunk:
                            yield chunk
                    elif line.startswith(b"\n"):
                        pass
                    else:
                        raise Exception(f"Unexpected line: {line}")
