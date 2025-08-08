from __future__ import annotations

import random
from typing import AsyncIterator

from .base_provider import AsyncAuthedProvider, ProviderModelMixin
from ..providers.helper import get_last_user_message
from ..requests import StreamSession, sse_stream, raise_for_status
from ..providers.response import AuthResult, TitleGeneration, JsonConversation, FinishReason
from ..typing import AsyncResult, Messages
from ..errors import MissingAuthError

class Kimi(AsyncAuthedProvider, ProviderModelMixin):
    url = "https://www.kimi.com"
    working = True
    active_by_default = True
    default_model = "kimi-k2"
    models = [default_model]
    model_aliases = {"moonshotai/Kimi-K2-Instruct": default_model}

    @classmethod
    async def on_auth_async(cls, proxy: str = None, **kwargs) -> AsyncIterator:
        device_id = str(random.randint(1000000000000000, 9999999999999999))
        async with StreamSession(proxy=proxy, impersonate="chrome") as session:
            async with session.post(
                "https://www.kimi.com/api/device/register",
                json={},
                headers={
                    "x-msh-device-id": device_id,
                    "x-msh-platform": "web",
                    "x-traffic-id": device_id
                }
            ) as response:
                await raise_for_status(response)
            data = await response.json()
            if not data.get("access_token"):
                raise Exception("No access token received")
        yield AuthResult(
            api_key=data.get("access_token"),
            device_id=device_id,
        )

    @classmethod
    async def create_authed(
        cls,
        model: str,
        messages: Messages,
        auth_result: AuthResult,
        proxy: str = None,
        conversation: JsonConversation = None,
        web_search: bool = False,
        **kwargs
    ) -> AsyncResult:
        async with StreamSession(
            proxy=proxy,
            impersonate="chrome",
            headers={
                "Authorization": f"Bearer {auth_result.api_key}",
            }
        ) as session:
            if conversation is None:
                async with session.post("https://www.kimi.com/api/chat", json={
                    "name":"未命名会话",
                    "born_from":"home",
                    "kimiplus_id":"kimi",
                    "is_example":False,
                    "source":"web",
                    "tags":[]
                }) as response:
                    try:
                        await raise_for_status(response)
                    except Exception as e:
                        if "匿名聊天使用次数超过" in str(e):
                            raise MissingAuthError("Anonymous chat usage limit exceeded")
                        raise e
                    chat_data = await response.json()
                conversation = JsonConversation(chat_id=chat_data.get("id"))
                yield conversation
            data = {
                "kimiplus_id": "kimi",
                "extend": {"sidebar": True},
                "model": "k2",
                "use_search": web_search,
                "messages": [
                    {
                        "role": "user",
                        "content": get_last_user_message(messages)
                    }
                ],
                "refs": [],
                "history": [],
                "scene_labels": [],
                "use_semantic_memory": False,
                "use_deep_research": False
            }
            async with session.post(
                f"https://www.kimi.com/api/chat/{conversation.chat_id}/completion/stream",
                json=data
            ) as response:
                await raise_for_status(response)
                async for line in sse_stream(response):
                    if line.get("event") == "cmpl":
                        yield line.get("text")
                    elif line.get("event") == "rename":
                        yield TitleGeneration(line.get("text"))
                    elif line.get("event") == "all_done":
                        yield FinishReason("stop")
                        break