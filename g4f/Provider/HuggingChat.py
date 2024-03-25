from __future__ import annotations

import json
import requests
from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt, get_connector


class HuggingChat(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://huggingface.co/chat"
    working = True
    default_model = "meta-llama/Llama-2-70b-chat-hf"
    models = [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "google/gemma-7b-it",
        "meta-llama/Llama-2-70b-chat-hf",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "codellama/CodeLlama-34b-Instruct-hf",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "openchat/openchat-3.5-0106",
    ]
    model_aliases = {
        "openchat/openchat_3.5": "openchat/openchat-3.5-0106",
    }

    @classmethod
    def get_models(cls):
        if not cls.models:
            url = f"{cls.url}/__data.json"
            data = requests.get(url).json()["nodes"][0]["data"]
            models = [data[key]["name"] for key in data[data[0]["models"]]]
            cls.models = [data[key] for key in models]
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        proxy: str = None,
        connector: BaseConnector = None,
        web_search: bool = False,
        cookies: dict = None,
        **kwargs
    ) -> AsyncResult:
        options = {"model": cls.get_model(model)}
        system_prompt = "\n".join([message["content"] for message in messages if message["role"] == "system"])
        if system_prompt:
            options["preprompt"] = system_prompt
            messages = [message for message in messages if message["role"] != "system"]
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
        }
        async with ClientSession(
            cookies=cookies,
            headers=headers,
            connector=get_connector(connector, proxy)
        ) as session:
            async with session.post(f"{cls.url}/conversation", json=options, proxy=proxy) as response:
                response.raise_for_status()
                conversation_id = (await response.json())["conversationId"]
            async with session.get(f"{cls.url}/conversation/{conversation_id}/__data.json") as response:
                response.raise_for_status()
                data: list = (await response.json())["nodes"][1]["data"]
                keys: list[int] = data[data[0]["messages"]]
                message_keys: dict = data[keys[0]]
                message_id: str = data[message_keys["id"]]
            options = {
                "id": message_id,
                "inputs": format_prompt(messages),
                "is_continue": False,
                "is_retry": False,
                "web_search": web_search
            }
            async with session.post(f"{cls.url}/conversation/{conversation_id}", json=options) as response:
                first_token = True
                async for line in response.content:
                    response.raise_for_status()
                    line = json.loads(line)
                    if "type" not in line:
                        raise RuntimeError(f"Response: {line}")
                    elif line["type"] == "stream":
                        token = line["token"]
                        if first_token:
                            token = token.lstrip()
                            first_token = False
                        yield token
                    elif line["type"] == "finalAnswer":
                        break
            async with session.delete(f"{cls.url}/conversation/{conversation_id}", proxy=proxy) as response:
                response.raise_for_status()
