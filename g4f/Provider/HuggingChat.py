from __future__ import annotations

import json
import requests
from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from ..providers.conversation import BaseConversation
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt, get_connector, get_cookies

class HuggingChat(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://huggingface.co/chat"
    working = True
    needs_auth = True
    default_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    models = [
        "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
        'CohereForAI/c4ai-command-r-plus',
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'google/gemma-1.1-7b-it',
        'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO',
        'mistralai/Mistral-7B-Instruct-v0.2',
        'meta-llama/Meta-Llama-3-70B-Instruct',
        'microsoft/Phi-3-mini-4k-instruct'
    ]
    model_aliases = {
        "mistralai/Mistral-7B-Instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.2"
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
        conversation: Conversation = None,
        return_conversation: bool = False,
        delete_conversation: bool = True,
        **kwargs
    ) -> AsyncResult:
        options = {"model": cls.get_model(model)}
        if cookies is None:
            cookies = get_cookies("huggingface.co", False)
        if return_conversation:
            delete_conversation = False

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
            if conversation is None:
                async with session.post(f"{cls.url}/conversation", json=options) as response:
                    await raise_for_status(response)
                    conversation_id = (await response.json())["conversationId"]
                if return_conversation:
                    yield Conversation(conversation_id)
            else:
                conversation_id = conversation.conversation_id
            async with session.get(f"{cls.url}/conversation/{conversation_id}/__data.json") as response:
                await raise_for_status(response)
                data: list = (await response.json())["nodes"][1]["data"]
                keys: list[int] = data[data[0]["messages"]]
                message_keys: dict = data[keys[0]]
                message_id: str = data[message_keys["id"]]
            options = {
                "id": message_id,
                "inputs": format_prompt(messages) if conversation is None else messages[-1]["content"],
                "is_continue": False,
                "is_retry": False,
                "web_search": web_search
            }
            async with session.post(f"{cls.url}/conversation/{conversation_id}", json=options) as response:
                first_token = True
                async for line in response.content:
                    await raise_for_status(response)
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
            if delete_conversation:
                async with session.delete(f"{cls.url}/conversation/{conversation_id}") as response:
                    await raise_for_status(response)

class Conversation(BaseConversation):
    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
