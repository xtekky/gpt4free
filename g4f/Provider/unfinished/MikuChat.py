from __future__ import annotations

import random, json
from datetime import datetime
from ...requests import StreamSession

from ...typing import AsyncGenerator
from ..base_provider import AsyncGeneratorProvider


class MikuChat(AsyncGeneratorProvider):
    url = "https://ai.okmiku.com"
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
            cls,
            model: str,
            messages: list[dict[str, str]],
            **kwargs
    ) -> AsyncGenerator:
        if not model:
            model = "gpt-3.5-turbo"
        headers = {
            "authority": "api.catgpt.cc",
            "accept": "application/json",
            "origin": cls.url,
            "referer": f"{cls.url}/chat/",
            'x-app-version': 'undefined',
            'x-date': get_datetime(),
            'x-fingerprint': get_fingerprint(),
            'x-platform': 'web'
        }
        async with StreamSession(headers=headers, impersonate="chrome107") as session:
            data = {
                "model": model,
                "top_p": 0.8,
                "temperature": 0.5,
                "presence_penalty": 1,
                "frequency_penalty": 0,
                "max_tokens": 2000,
                "stream": True,
                "messages": messages,
            }
            async with session.post("https://api.catgpt.cc/ai/v1/chat/completions", json=data) as response:
                print(await response.text())
                response.raise_for_status()
                async for line in response.iter_lines():
                    if line.startswith(b"data: "):
                        line = json.loads(line[6:])
                        chunk = line["choices"][0]["delta"].get("content")
                        if chunk:
                            yield chunk

def k(e: str, t: int):
    a = len(e) & 3
    s = len(e) - a
    i = t
    c = 3432918353
    o = 461845907
    n = 0
    r = 0
    while n < s:
        r = (ord(e[n]) & 255) | ((ord(e[n + 1]) & 255) << 8) | ((ord(e[n + 2]) & 255) << 16) | ((ord(e[n + 3]) & 255) << 24)
        n += 4
        r = (r & 65535) * c + (((r >> 16) * c & 65535) << 16) & 4294967295
        r = (r << 15) | (r >> 17)
        r = (r & 65535) * o + (((r >> 16) * o & 65535) << 16) & 4294967295
        i ^= r
        i = (i << 13) | (i >> 19)
        l = (i & 65535) * 5 + (((i >> 16) * 5 & 65535) << 16) & 4294967295
        i = (l & 65535) + 27492 + (((l >> 16) + 58964 & 65535) << 16)
    
    if a == 3:
        r ^= (ord(e[n + 2]) & 255) << 16
    elif a == 2:
        r ^= (ord(e[n + 1]) & 255) << 8
    elif a == 1:
        r ^= ord(e[n]) & 255
        r = (r & 65535) * c + (((r >> 16) * c & 65535) << 16) & 4294967295
        r = (r << 15) | (r >> 17)
        r = (r & 65535) * o + (((r >> 16) * o & 65535) << 16) & 4294967295
        i ^= r

    i ^= len(e)
    i ^= i >> 16
    i = (i & 65535) * 2246822507 + (((i >> 16) * 2246822507 & 65535) << 16) & 4294967295
    i ^= i >> 13
    i = (i & 65535) * 3266489909 + (((i >> 16) * 3266489909 & 65535) << 16) & 4294967295
    i ^= i >> 16
    return i & 0xFFFFFFFF

def get_fingerprint() -> str:
    return str(k(str(int(random.random() * 100000)), 256))

def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")