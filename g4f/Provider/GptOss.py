from __future__ import annotations


from ..typing import AsyncResult, Messages, MediaListType
from ..providers.response import JsonConversation, Reasoning, TitleGeneration
from ..requests import StreamSession, raise_for_status
from ..config import DEFAULT_MODEL
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_last_user_message

class GptOss(AsyncGeneratorProvider, ProviderModelMixin):
    label = "gpt-oss (playground)"
    url = "https://gpt-oss.com"
    api_endpoint = "https://api.gpt-oss.com/chatkit"
    working = True
    active_by_default = True
    
    default_model = "gpt-oss-120b"
    models = [default_model, "gpt-oss-20b"]
    model_aliases = {
        DEFAULT_MODEL: default_model,
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        media: MediaListType = None,
        conversation: JsonConversation = None,
        reasoning_effort: str = "high",
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        if media:
            raise ValueError("Media is not supported by gpt-oss")
        model = cls.get_model(model)
        user_message = get_last_user_message(messages)
        cookies = {}
        if conversation is None:
            data = {
                "op": "threads.create",
                "params": {
                    "input": {
                        "text": user_message,
                        "content": [{"type": "input_text", "text": user_message}],
                        "quoted_text": "",
                        "attachments": []
                    }
                }
            }
        else:
            data = {
                "op":"threads.addMessage",
                "params": {
                    "input": {
                        "text": user_message,
                        "content": [{"type": "input_text", "text": user_message}],
                        "quoted_text": "",
                        "attachments": []
                    },
                    "threadId": conversation.id
                }
            }
            cookies["user_id"] = conversation.user_id
        headers =  {
            "accept": "text/event-stream",
            "x-reasoning-effort": reasoning_effort,
            "x-selected-model": model,
            "x-show-reasoning": "true"
        }
        async with StreamSession(
            headers=headers,
            cookies=cookies,
            proxy=proxy,
        ) as session:
            async with session.post(
                cls.api_endpoint,
                json=data
            ) as response:
                await raise_for_status(response)
                async for chunk in response.sse():
                    if chunk.get("type") == "thread.created":
                        yield JsonConversation(id=chunk["thread"]["id"], user_id=response.cookies.get("user_id"))
                    elif chunk.get("type") == "thread.item_updated":
                        entry = chunk.get("update", {}).get("entry", chunk.get("update", {}))
                        if entry.get("type") == "thought":
                            yield Reasoning(entry.get("content"))
                        elif entry.get("type") == "recap":
                            pass #yield Reasoning(status=entry.get("summary"))
                        elif entry.get("type") == "assistant_message.content_part.text_delta":
                            yield entry.get("delta")
                    elif chunk.get("type") == "thread.updated":
                        yield TitleGeneration(chunk["thread"]["title"])