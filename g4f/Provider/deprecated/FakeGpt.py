from __future__ import annotations

import uuid, time, random, json
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider
from ..helper import format_prompt, get_random_string


class FakeGpt(AsyncGeneratorProvider):
    url                   = "https://chat-shared2.zhile.io"
    supports_gpt_35_turbo = True
    working               = False
    _access_token         = None
    _cookie_jar           = None

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "Accept-Language": "en-US",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
            "Referer": "https://chat-shared2.zhile.io/?v=2",
            "sec-ch-ua": '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
            "sec-ch-ua-platform": '"Linux"',
            "sec-ch-ua-mobile": "?0",
        }
        async with ClientSession(headers=headers, cookie_jar=cls._cookie_jar) as session:
            if not cls._access_token:
                async with session.get(f"{cls.url}/api/loads", params={"t": int(time.time())}, proxy=proxy) as response:
                    response.raise_for_status()
                    list = (await response.json())["loads"]
                    token_ids = [t["token_id"] for t in list]
                data = {
                    "token_key": random.choice(token_ids),
                    "session_password": get_random_string()
                }
                async with session.post(f"{cls.url}/auth/login", data=data, proxy=proxy) as response:
                    response.raise_for_status()
                async with session.get(f"{cls.url}/api/auth/session", proxy=proxy) as response:
                    response.raise_for_status()
                    cls._access_token = (await response.json())["accessToken"]
                    cls._cookie_jar = session.cookie_jar
            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "X-Authorization": f"Bearer {cls._access_token}",
            }
            prompt = format_prompt(messages)
            data = {
                "action": "next",
                "messages": [
                    {
                        "id": str(uuid.uuid4()),
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": [prompt]},
                        "metadata": {},
                    }
                ],
                "parent_message_id": str(uuid.uuid4()),
                "model": "text-davinci-002-render-sha",
                "plugin_ids": [],
                "timezone_offset_min": -120,
                "suggestions": [],
                "history_and_training_disabled": True,
                "arkose_token": "",
                "force_paragen": False,
            }
            last_message = ""
            async with session.post(f"{cls.url}/api/conversation", json=data, headers=headers, proxy=proxy) as response:
                async for line in response.content:
                    if line.startswith(b"data: "):
                        line = line[6:]
                        if line == b"[DONE]":
                            break
                        try:
                            line = json.loads(line)
                            if line["message"]["metadata"]["message_type"] == "next":
                                new_message = line["message"]["content"]["parts"][0]
                                yield new_message[len(last_message):]
                                last_message = new_message
                        except:
                            continue
            if not last_message:
                raise RuntimeError("No valid response")