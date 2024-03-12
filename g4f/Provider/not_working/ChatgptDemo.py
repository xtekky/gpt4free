from __future__ import annotations

import time, json, re, asyncio
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ...errors import RateLimitError
from ..base_provider import AsyncGeneratorProvider
from ..helper import format_prompt

class ChatgptDemo(AsyncGeneratorProvider):
    url = "https://chatgptdemo.info/chat"
    working = False
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "authority": "chatgptdemo.info",
            "accept-language": "en-US",
            "origin": "https://chatgptdemo.info",
            "referer": "https://chatgptdemo.info/chat/",
            "sec-ch-ua": '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        }
        async with ClientSession(headers=headers) as session:
            async with session.get(f"{cls.url}/", proxy=proxy) as response:
                response.raise_for_status()
                text = await response.text()
            result = re.search(
                r'<div id="USERID" style="display: none">(.*?)<\/div>',
                text,
            )
            if result:
                user_id = result.group(1)
            else:
                raise RuntimeError("No user id found")
            async with session.post(f"https://chatgptdemo.info/chat/new_chat", json={"user_id": user_id}, proxy=proxy) as response:
                response.raise_for_status()
                chat_id = (await response.json())["id_"]
            if not chat_id:
                raise RuntimeError("Could not create new chat")
            await asyncio.sleep(10)
            data = {
                "question": format_prompt(messages),
                "chat_id": chat_id,
                "timestamp": int((time.time())*1e3),
            }
            async with session.post(f"https://chatgptdemo.info/chat/chat_api_stream", json=data, proxy=proxy) as response:
                if response.status == 429:
                    raise RateLimitError("Rate limit reached")
                response.raise_for_status()
                async for line in response.content:
                    if line.startswith(b"data: "):
                        line = json.loads(line[6:-1])

                        chunk = line["choices"][0]["delta"].get("content")
                        if chunk:
                            yield chunk