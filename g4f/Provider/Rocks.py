import asyncio
import json
from aiohttp import ClientSession
from ..typing import Messages, AsyncResult
from .base_provider import AsyncGeneratorProvider

class Rocks(AsyncGeneratorProvider):
    url = "https://api.airforce"
    api_endpoint = "/chat/completions"
    supports_message_history = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    supports_stream = True
    supports_system_message = True
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
            "Authorization": "Bearer missing api key",
            "Origin": "https://llmplayground.net",
            "Referer": "https://llmplayground.net/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        }

        async with ClientSession() as session:
            async with session.post(
                f"{cls.url}{cls.api_endpoint}",
                json=payload,
                proxy=proxy,
                headers=headers
            ) as response:
                response.raise_for_status()
                last_chunk_time = asyncio.get_event_loop().time()
                
                async for line in response.content:
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_chunk_time > 5:
                        return
                    
                    if line.startswith(b"\n"):
                        pass
                    elif "discord.com/invite/" in line.decode() or "discord.gg/" in line.decode():
                        pass # trolled
                    elif line.startswith(b"data: "):
                        try:
                            line = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        chunk = line["choices"][0]["delta"].get("content")
                        if chunk:
                            yield chunk
                            last_chunk_time = current_time
                    else:
                        raise Exception(f"Unexpected line: {line}")
                return