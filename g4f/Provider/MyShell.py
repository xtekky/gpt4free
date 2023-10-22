from __future__ import annotations

import time, random, json

from ..requests import StreamSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
from .helper import format_prompt

class MyShell(AsyncGeneratorProvider):
    url = "https://app.myshell.ai/chat"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        **kwargs
    ) -> AsyncResult:
        user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
        headers = {
            "User-Agent": user_agent,
            "Myshell-Service-Name": "organics-api",
            "Visitor-Id": generate_visitor_id(user_agent)
        }
        async with StreamSession(
            impersonate="chrome107",
            proxies={"https": proxy},
            timeout=timeout,
            headers=headers
        ) as session:
            prompt = format_prompt(messages)
            data = {
                "botId": "1",
                "conversation_scenario": 3,
                "message": prompt,
                "messageType": 1
            }
            async with session.post("https://api.myshell.ai/v1/bot/chat/send_message", json=data) as response:
                response.raise_for_status()
                event = None
                async for line in response.iter_lines():
                    if line.startswith(b"event: "):
                        event = line[7:]
                    elif event == b"MESSAGE_REPLY_SSE_ELEMENT_EVENT_NAME_TEXT":
                        if line.startswith(b"data: "):
                            yield json.loads(line[6:])["content"]
                    if event == b"MESSAGE_REPLY_SSE_ELEMENT_EVENT_NAME_TEXT_STREAM_PUSH_FINISHED":
                        break


def xor_hash(B: str):
    r = []
    i = 0
    
    def o(e, t):
        o_val = 0
        for i in range(len(t)):
            o_val |= r[i] << (8 * i)
        return e ^ o_val
    
    for e in range(len(B)):
        t = ord(B[e])
        r.insert(0, 255 & t)
        
        if len(r) >= 4:
            i = o(i, r)
            r = []
    
    if len(r) > 0:
        i = o(i, r)
    
    return hex(i)[2:]

def performance() -> str:
    t = int(time.time() * 1000)
    e = 0
    while t == int(time.time() * 1000):
        e += 1
    return hex(t)[2:] + hex(e)[2:]

def generate_visitor_id(user_agent: str) -> str:
    f = performance()
    r = hex(int(random.random() * (16**16)))[2:-2]
    d = xor_hash(user_agent)
    e = hex(1080 * 1920)[2:]
    return f"{f}-{r}-{d}-{e}-{f}"