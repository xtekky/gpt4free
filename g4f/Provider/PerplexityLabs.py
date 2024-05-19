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
    default_model = "mixtral-8x7b-instruct"
    models = [
        "llama-3-sonar-large-32k-online", "llama-3-sonar-small-32k-online", "llama-3-sonar-large-32k-chat", "llama-3-sonar-small-32k-chat",
        "dbrx-instruct", "claude-3-haiku-20240307", "llama-3-8b-instruct", "llama-3-70b-instruct", "codellama-70b-instruct", "mistral-7b-instruct",
        "llava-v1.5-7b-wrapper", "llava-v1.6-34b", "mixtral-8x7b-instruct", "mixtral-8x22b-instruct", "mistral-medium", "gemma-2b-it", "gemma-7b-it",
        "related"
    ]
    model_aliases = {
        "mistralai/Mistral-7B-Instruct-v0.1": "mistral-7b-instruct",
        "mistralai/Mistral-7B-Instruct-v0.2": "mistral-7b-instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "mixtral-8x7b-instruct",
        "codellama/CodeLlama-70b-Instruct-hf": "codellama-70b-instruct",
        "llava-v1.5-7b": "llava-v1.5-7b-wrapper",
        "databricks/dbrx-instruct": "dbrx-instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct": "llama-3-70b-instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct": "llama-3-8b-instruct"
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
