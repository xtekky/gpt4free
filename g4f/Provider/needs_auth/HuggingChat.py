from __future__ import annotations

import json, uuid

from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider
from ..helper import format_prompt, get_cookies


class HuggingChat(AsyncGeneratorProvider):
    url = "https://huggingface.co/chat"
    needs_auth = True
    working = True
    model = "meta-llama/Llama-2-70b-chat-hf"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        proxy: str = None,
        cookies: dict = None,
        **kwargs
    ) -> AsyncResult:
        model = model if model else cls.model
        if proxy and "://" not in proxy:
            proxy = f"http://{proxy}"
        if not cookies:
            cookies = get_cookies(".huggingface.co")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
        }
        async with ClientSession(
            cookies=cookies,
            headers=headers
        ) as session:
            async with session.post(f"{cls.url}/conversation", json={"model": model}, proxy=proxy) as response:
                conversation_id = (await response.json())["conversationId"]

            send = {
                "id": str(uuid.uuid4()),
                "inputs": format_prompt(messages),
                "is_retry": False,
                "response_id": str(uuid.uuid4()),
                "web_search": False
            }
            async with session.post(f"{cls.url}/conversation/{conversation_id}", json=send, proxy=proxy) as response:
                async for line in response.content:
                    line = json.loads(line[:-1])
                    if "type" not in line:
                        raise RuntimeError(f"Response: {line}")
                    elif line["type"] == "stream":
                        yield line["token"]
                    elif line["type"] == "finalAnswer":
                        break
                
            async with session.delete(f"{cls.url}/conversation/{conversation_id}", proxy=proxy) as response:
                response.raise_for_status()


    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("proxy", "str"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
