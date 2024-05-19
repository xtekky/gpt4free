from __future__ import annotations

import json
import aiohttp

from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_connector
from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from ..providers.conversation import BaseConversation

class DuckDuckGo(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://duckduckgo.com/duckchat"
    working = True
    supports_gpt_35_turbo = True
    supports_message_history = True

    default_model = "gpt-3.5-turbo-0125"
    models = ["gpt-3.5-turbo-0125", "claude-3-haiku-20240307"]
    model_aliases = {
        "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
        "claude-3-haiku": "claude-3-haiku-20240307"
    }

    status_url = "https://duckduckgo.com/duckchat/v1/status"
    chat_url = "https://duckduckgo.com/duckchat/v1/chat"
    user_agent = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0'
    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/event-stream',
        'Accept-Language': 'de,en-US;q=0.7,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://duckduckgo.com/',
        'Content-Type': 'application/json',
        'Origin': 'https://duckduckgo.com',
        'Connection': 'keep-alive',
        'Cookie': 'dcm=1',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Pragma': 'no-cache',
        'TE': 'trailers'
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        connector: aiohttp.BaseConnector = None,
        conversation: Conversation = None,
        return_conversation: bool = False,
        **kwargs
    ) -> AsyncResult:
        async with aiohttp.ClientSession(headers=cls.headers, connector=get_connector(connector, proxy)) as session:
            if conversation is not None and len(messages) > 1:
                vqd_4 = conversation.vqd_4
                messages = [*conversation.messages, messages[-2], messages[-1]]
            else:
                async with session.get(cls.status_url, headers={"x-vqd-accept": "1"}) as response:
                    await raise_for_status(response)
                    vqd_4 = response.headers.get("x-vqd-4")
                messages = [messages[-1]]
            payload = {
                'model': cls.get_model(model),
                'messages': messages
            }
            async with session.post(cls.chat_url, json=payload, headers={"x-vqd-4": vqd_4}) as response:
                await raise_for_status(response)
                if return_conversation:
                    yield Conversation(response.headers.get("x-vqd-4"), messages)
                async for line in response.content:
                    if line.startswith(b"data: "):
                        chunk = line[6:]
                        if chunk.startswith(b"[DONE]"):
                            break
                        data = json.loads(chunk)
                        if "message" in data and data["message"]:
                            yield data["message"]

class Conversation(BaseConversation):
    def __init__(self, vqd_4: str, messages: Messages) -> None:
        self.vqd_4 = vqd_4
        self.messages = messages