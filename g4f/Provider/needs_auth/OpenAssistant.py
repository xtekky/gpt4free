from __future__ import annotations

import json

from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider
from ..helper import format_prompt, get_cookies


class OpenAssistant(AsyncGeneratorProvider):
    url = "https://open-assistant.io/chat"
    needs_auth = True
    working = False
    model = "OA_SFT_Llama_30B_6"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        cookies: dict = None,
        **kwargs
    ) -> AsyncResult:
        if not cookies:
            cookies = get_cookies("open-assistant.io")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
        }
        async with ClientSession(
            cookies=cookies,
            headers=headers
        ) as session:
            async with session.post("https://open-assistant.io/api/chat", proxy=proxy) as response:
                chat_id = (await response.json())["id"]

            data = {
                "chat_id": chat_id,
                "content": f"<s>[INST]\n{format_prompt(messages)}\n[/INST]",
                "parent_id": None
            }
            async with session.post("https://open-assistant.io/api/chat/prompter_message", proxy=proxy, json=data) as response:
                parent_id = (await response.json())["id"]

            data = {
                "chat_id": chat_id,
                "parent_id": parent_id,
                "model_config_name": model if model else cls.model,
                "sampling_parameters":{
                    "top_k": 50,
                    "top_p": None,
                    "typical_p": None,
                    "temperature": 0.35,
                    "repetition_penalty": 1.1111111111111112,
                    "max_new_tokens": 1024,
                    **kwargs
                },
                "plugins":[]
            }
            async with session.post("https://open-assistant.io/api/chat/assistant_message", proxy=proxy, json=data) as response:
                data = await response.json()
                if "id" in  data:
                    message_id = data["id"]
                elif "message" in data:
                    raise RuntimeError(data["message"])
                else:
                    response.raise_for_status()
            
            params = {
                'chat_id': chat_id,
                'message_id': message_id,
            }
            async with session.post("https://open-assistant.io/api/chat/events", proxy=proxy, params=params) as response:
                start = "data: "
                async for line in response.content:
                    line = line.decode("utf-8")
                    if line and line.startswith(start):
                        line = json.loads(line[len(start):])
                        if line["event_type"] == "token":
                            yield line["text"]

            params = {
                'chat_id': chat_id,
            }
            async with session.delete("https://open-assistant.io/api/chat", proxy=proxy, params=params) as response:
                response.raise_for_status()
