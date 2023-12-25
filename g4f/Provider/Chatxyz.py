from __future__ import annotations
from aiohttp import ClientSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider
import json
class HogeService(AsyncGeneratorProvider):
    url                   = "https://chat.3211000.xyz/api/openai/v1/chat/completions"
    working               = True
    supports_gpt_35_turbo = True

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = "gpt-3.5-turbo"
        headers = {
            'Accept': 'text/event-stream',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.5',
            'Alt-Used': 'chat.3211000.xyz',
            'Content-Type': 'application/json',
            'Host': 'chat.3211000.xyz',
            'Origin': 'https://chat.3211000.xyz',
            'Referer': 'https://chat.3211000.xyz/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'TE': 'trailers',
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'x-requested-with': 'XMLHttpRequest'
        }   
        async with ClientSession(headers=headers) as session:
            data = {
                "messages": messages,
                "stream": True,
                "model": "gpt-3.5-turbo",
                "temperature": 0.5,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "top_p": 1
            }    

            async with session.post(cls.url,json=data) as response:
                async for chunk in response.content:
                    line = chunk.decode() 
                    if line.startswith("data: [DONE]"):
                            break
                    elif line.startswith("data: "):
                            line = json.loads(line[6:])
                            if(line["choices"][0]["delta"]["content"]!=None):
                                yield line["choices"][0]["delta"]["content"]