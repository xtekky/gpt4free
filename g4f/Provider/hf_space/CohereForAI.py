from __future__ import annotations

import json
from aiohttp import ClientSession, FormData

from ...typing import AsyncResult, Messages
from ...requests import raise_for_status
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, get_last_user_message
from ...providers.response import JsonConversation, TitleGeneration

class CohereForAI(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://cohereforai-c4ai-command.hf.space"
    conversation_url = f"{url}/conversation"

    working = True

    default_model = "command-r-plus-08-2024"
    models = [
        default_model,
        "command-r-08-2024",
        "command-r-plus",
        "command-r",
        "command-r7b-12-2024",
    ]
    model_aliases = {
        "command-r-plus": "command-r-plus-08-2024",
        "command-r": "command-r-08-2024",
        "command-r7b": "command-r7b-12-2024",
    }

    @classmethod
    async def create_async_generator(
        cls, model: str, messages: Messages,
        api_key: str = None, 
        proxy: str = None,
        conversation: JsonConversation = None,
        return_conversation: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        headers = {
            "Origin": cls.url,
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://cohereforai-c4ai-command.hf.space/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Priority": "u=4",
        }
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        async with ClientSession(
            headers=headers,
            cookies=None if conversation is None else conversation.cookies
        ) as session:
            system_prompt = "\n".join([message["content"] for message in messages if message["role"] == "system"])
            messages = [message for message in messages if message["role"] != "system"]
            inputs = format_prompt(messages) if conversation is None else get_last_user_message(messages)
            if conversation is None or conversation.model != model or conversation.preprompt != system_prompt:
                data = {"model": model, "preprompt": system_prompt}
                async with session.post(cls.conversation_url, json=data, proxy=proxy) as response:
                    await raise_for_status(response)
                    conversation = JsonConversation(
                        **await response.json(),
                        **data,
                        cookies={n: c.value for n, c in response.cookies.items()}
                    )
                    if return_conversation:
                        yield conversation
            async with session.get(f"{cls.conversation_url}/{conversation.conversationId}/__data.json?x-sveltekit-invalidated=11", proxy=proxy) as response:
                await raise_for_status(response)
                node = json.loads((await response.text()).splitlines()[0])["nodes"][1]
                if node["type"] == "error":
                    raise RuntimeError(node["error"])
                data = node["data"]
                message_id = data[data[data[data[0]["messages"]][-1]]["id"]]
            data = FormData()
            data.add_field(
                "data",
                json.dumps({"inputs": inputs, "id": message_id, "is_retry": False, "is_continue": False, "web_search": False, "tools": []}),
                content_type="application/json"
            )
            async with session.post(f"{cls.conversation_url}/{conversation.conversationId}", data=data, proxy=proxy) as response:
                await raise_for_status(response)
                async for chunk in response.content:
                    try:
                        data = json.loads(chunk)
                    except (json.JSONDecodeError) as e:
                        raise RuntimeError(f"Failed to read response: {chunk.decode(errors='replace')}", e)
                    if data["type"] == "stream":
                        yield data["token"].replace("\u0000", "")
                    elif data["type"] == "title":
                        yield TitleGeneration(data["title"])
                    elif data["type"] == "finalAnswer":
                        break