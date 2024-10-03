from __future__ import annotations

import json
import aiohttp
import asyncio
from typing import Optional
import base64

from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_connector
from ..errors import ResponseError
from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from ..providers.conversation import BaseConversation


class DDG(AsyncGeneratorProvider, ProviderModelMixin):
    url = base64.b64decode("aHR0cHM6Ly9kdWNrZHVja2dvLmNvbS9haWNoYXQ=").decode("utf-8")
    working = True
    supports_gpt_35_turbo = True
    supports_message_history = True

    default_model = "gpt-4o-mini"
    models = ["gpt-4o-mini", "claude-3-haiku-20240307", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
    model_aliases = {
        "gpt-4": "gpt-4o-mini",
        "gpt-4o": "gpt-4o-mini",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1"
    }

    # Obfuscated URLs and headers
    status_url = base64.b64decode("aHR0cHM6Ly9kdWNrZHVja2dvLmNvbS9kdWNrY2hhdC92MS9zdGF0dXM=").decode("utf-8")
    chat_url = base64.b64decode("aHR0cHM6Ly9kdWNrZHVja2dvLmNvbS9kdWNrY2hhdC92MS9jaGF0").decode("utf-8")
    referer = base64.b64decode("aHR0cHM6Ly9kdWNrZHVja2dvLmNvbS8=").decode("utf-8")
    origin = base64.b64decode("aHR0cHM6Ly9kdWNrZHVja2dvLmNvbQ==").decode("utf-8")

    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'
    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/event-stream',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Referer': referer,
        'Content-Type': 'application/json',
        'Origin': origin,
        'Connection': 'keep-alive',
        'Cookie': 'dcm=3',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Pragma': 'no-cache',
        'TE': 'trailers'
    }

    @classmethod
    async def get_vqd(cls, session: aiohttp.ClientSession) -> Optional[str]:
        try:
            async with session.get(cls.status_url, headers={"x-vqd-accept": "1"}) as response:
                await raise_for_status(response)
                return response.headers.get("x-vqd-4")
        except Exception as e:
            print(f"Error getting VQD: {e}")
            return None

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
            if len(messages) > 1:
                if conversation is None:
                    raise ValueError(f"More than one message requires using conversation.")

                vqd_4 = conversation.vqd_4
            else:
                for _ in range(3):  # Try up to 3 times to get a valid VQD
                    vqd_4 = await cls.get_vqd(session)
                    if vqd_4:
                        break
                    await asyncio.sleep(1)  # Wait a bit before retrying

                if not vqd_4:
                    raise Exception("Failed to obtain a valid VQD token")

            payload = {
                'model': cls.get_model(model),
                'messages': [{'role': m['role'], 'content': m['content']} for m in messages]
            }

            async with session.post(cls.chat_url, json=payload, headers={"x-vqd-4": vqd_4}) as response:
                if response.status == 400:
                    if "ERR_INVALID_VQD" in await response.text():
                        raise ResponseError(
                            f"Status {response.status}: Conversation does not exactly match the messages or its id (vqd-4) is invalid"
                        )
                await raise_for_status(response)
                if return_conversation:
                    vqd_4_new = response.headers.get("x-vqd-4")
                    if vqd_4_new is not None:
                        yield Conversation(vqd_4_new, messages)
                    elif conversation is not None:
                        raise ResponseError(
                            "Conversation requested, but is not present in the response"
                        )

                async for line in response.content:
                    if line.startswith(b"data: "):
                        chunk = line[6:]
                        if chunk.startswith(b"[DONE]"):
                            break
                        try:
                            data = json.loads(chunk)
                            if "message" in data and data["message"]:
                                yield data["message"]
                        except json.JSONDecodeError:
                            print(f"Failed to decode JSON: {chunk}")

class Conversation(BaseConversation):
    def __init__(self, vqd_4: str, messages: Messages) -> None:
        self.vqd_4 = vqd_4
