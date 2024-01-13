from __future__ import annotations

import json
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider



models = {
     "claude-v1":"claude-2.1",
     "claude-v2":"claude-2.0",
     "gpt_35_turbo":"gpt-3.5-turbo-1106",
     "gpt-4":"gpt-4",
     "gemini-pro":"google-gemini-pro"
}

class FreeChatgpt(AsyncGeneratorProvider):
    url = "https://free.chatgpt.org.uk"
    working = True
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    supports_message_history = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = models[model] if model in models else "gpt-3.5-turbo-1106"
        headers = {
    "Accept": "application/json, text/event-stream",
    "Content-Type":"application/json",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.5",
    "Host":"free.chatgpt.org.uk",
    "Referer":f"{cls.url}/",
    "Origin":f"{cls.url}",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", 
}
        async with ClientSession(headers=headers) as session:
            data = {"messages":messages,"stream":True,"model":model,"temperature":0.5,"presence_penalty":0,"frequency_penalty":0,"top_p":1}
            async with session.post(f'{cls.url}/api/openai/v1/chat/completions',json=data) as result:
                async for chunk in result.content:
                    
                    line = chunk.decode()
                    if line.startswith("data: [DONE]"):
                            break
                    elif line.startswith("data: "):
                         line = json.loads(line[6:])
                         if(line["choices"]==[]):
                              continue
                         if(line["choices"][0]["delta"].get("content") and line["choices"][0]["delta"]["content"]!=None):
                              yield line["choices"][0]["delta"]["content"]