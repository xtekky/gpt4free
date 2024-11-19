from __future__ import annotations

import random
import json

from ..typing import AsyncResult, Messages
from ..requests import StreamSession, raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

API_URL = "https://www.perplexity.ai/socket.io/"
WS_URL = "wss://www.perplexity.ai/socket.io/"

class PerplexityLabs(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://labs.perplexity.ai"
    working = True
    default_model = "llama-3.1-70b-instruct"
    models = [
        "llama-3.1-sonar-large-128k-online",
        "llama-3.1-sonar-small-128k-online",
        "llama-3.1-sonar-large-128k-chat",
        "llama-3.1-sonar-small-128k-chat",
        "llama-3.1-8b-instruct",
        "llama-3.1-70b-instruct",
        "/models/LiquidCloud",
    ]
    
    model_aliases = {
        "sonar-online": "llama-3.1-sonar-large-128k-online",
        "sonar-online": "sonar-small-128k-online",
        "sonar-chat": "llama-3.1-sonar-large-128k-chat",
        "sonar-chat": "llama-3.1-sonar-small-128k-chat",
        "llama-3.1-8b": "llama-3.1-8b-instruct",
        "llama-3.1-70b": "llama-3.1-70b-instruct",
        "lfm-40b": "/models/LiquidCloud",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Accept": "*/*",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Origin": cls.url,
            "Connection": "keep-alive",
            "Referer": f"{cls.url}/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "TE": "trailers",
        }
        async with StreamSession(headers=headers, proxies={"all": proxy}) as session:
            t = format(random.getrandbits(32), "08x")
            async with session.get(
                f"{API_URL}?EIO=4&transport=polling&t={t}"
            ) as response:
                await raise_for_status(response)
                text = await response.text()
            assert text.startswith("0")
            sid = json.loads(text[1:])["sid"]
            post_data = '40{"jwt":"anonymous-ask-user"}'
            async with session.post(
                f"{API_URL}?EIO=4&transport=polling&t={t}&sid={sid}",
                data=post_data
            ) as response:
                await raise_for_status(response)
                assert await response.text() == "OK"
            async with session.ws_connect(f"{WS_URL}?EIO=4&transport=websocket&sid={sid}", autoping=False) as ws:
                await ws.send_str("2probe")
                assert(await ws.receive_str() == "3probe")
                await ws.send_str("5")
                assert(await ws.receive_str())
                assert(await ws.receive_str() == "6")
                message_data = {
                    "version": "2.5",
                    "source": "default",
                    "model": cls.get_model(model),
                    "messages": messages
                }
                await ws.send_str("42" + json.dumps(["perplexity_labs", message_data]))
                last_message = 0
                while True:
                    message = await ws.receive_str()
                    if message == "2":
                        if last_message == 0:
                            raise RuntimeError("Unknown error")
                        await ws.send_str("3")
                        continue
                    try:
                        data = json.loads(message[2:])[1]
                        yield data["output"][last_message:]
                        last_message = len(data["output"])
                        if data["final"]:
                            break
                    except:
                        raise RuntimeError(f"Message: {message}")
