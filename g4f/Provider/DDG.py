from __future__ import annotations

import json
import aiohttp
from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin, BaseConversation
from .helper import format_prompt
from ..requests.aiohttp import get_connector
from ..requests.raise_for_status import raise_for_status
from .. import debug

MODELS = [
    {"model":"gpt-4o","modelName":"GPT-4o","modelVariant":None,"modelStyleId":"gpt-4o-mini","createdBy":"OpenAI","moderationLevel":"HIGH","isAvailable":1,"inputCharLimit":16e3,"settingId":"4"},
    {"model":"gpt-4o-mini","modelName":"GPT-4o","modelVariant":"mini","modelStyleId":"gpt-4o-mini","createdBy":"OpenAI","moderationLevel":"HIGH","isAvailable":0,"inputCharLimit":16e3,"settingId":"3"},
    {"model":"claude-3-5-sonnet-20240620","modelName":"Claude 3.5","modelVariant":"Sonnet","modelStyleId":"claude-3-haiku","createdBy":"Anthropic","moderationLevel":"HIGH","isAvailable":1,"inputCharLimit":16e3,"settingId":"7"},
    {"model":"claude-3-opus-20240229","modelName":"Claude 3","modelVariant":"Opus","modelStyleId":"claude-3-haiku","createdBy":"Anthropic","moderationLevel":"HIGH","isAvailable":1,"inputCharLimit":16e3,"settingId":"2"},
    {"model":"claude-3-haiku-20240307","modelName":"Claude 3","modelVariant":"Haiku","modelStyleId":"claude-3-haiku","createdBy":"Anthropic","moderationLevel":"HIGH","isAvailable":0,"inputCharLimit":16e3,"settingId":"1"},
    {"model":"meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo","modelName":"Llama 3.1","modelVariant":"70B","modelStyleId":"llama-3","createdBy":"Meta","moderationLevel":"MEDIUM","isAvailable":0,"isOpenSource":0,"inputCharLimit":16e3,"settingId":"5"},
    {"model":"mistralai/Mixtral-8x7B-Instruct-v0.1","modelName":"Mixtral","modelVariant":"8x7B","modelStyleId":"mixtral","createdBy":"Mistral AI","moderationLevel":"LOW","isAvailable":0,"isOpenSource":0,"inputCharLimit":16e3,"settingId":"6"}
]

class Conversation(BaseConversation):
    vqd: str = None
    message_history: Messages = []

    def __init__(self, model: str):
        self.model = model

class DDG(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://duckduckgo.com"
    api_endpoint = "https://duckduckgo.com/duckchat/v1/chat"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True

    default_model = "gpt-4o-mini"
    models = [model.get("model") for model in MODELS]
    model_aliases = {
        "claude-3-haiku": "claude-3-haiku-20240307",
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "gpt-4": "gpt-4o-mini"
    }

    @classmethod
    async def get_vqd(cls, proxy: str, connector: BaseConnector = None):
        status_url = "https://duckduckgo.com/duckchat/v1/status"
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'Accept': 'text/event-stream',
            'x-vqd-accept': '1'
        }
        async with aiohttp.ClientSession(connector=get_connector(connector, proxy)) as session:
            async with session.get(status_url, headers=headers) as response:
                await raise_for_status(response)
                return response.headers.get("x-vqd-4")

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        conversation: Conversation = None,
        return_conversation: bool = False,
        proxy: str = None,
        connector: BaseConnector = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)

        is_new_conversation = False
        if conversation is None:
            conversation = Conversation(model)
            is_new_conversation = True
        debug.last_model = model
        if conversation.vqd is None:
            conversation.vqd = await cls.get_vqd(proxy, connector)
        if not conversation.vqd:
            raise Exception("Failed to obtain VQD token")

        headers = {
            'accept': 'text/event-stream',
            'content-type': 'application/json',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'x-vqd-4': conversation.vqd,
        }
        async with ClientSession(headers=headers, connector=get_connector(connector, proxy)) as session:
            if is_new_conversation:
                conversation.message_history = [{"role": "user", "content": format_prompt(messages)}]
            else:
                conversation.message_history = [
                    *conversation.message_history,
                    messages[-2],
                    messages[-1]
                ]
            if return_conversation:
                yield conversation
            data = {
                "model": conversation.model,
                "messages": conversation.message_history
            }
            async with session.post(cls.api_endpoint, json=data) as response:
                conversation.vqd = response.headers.get("x-vqd-4")
                await raise_for_status(response)
                async for line in response.content:
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            json_str = decoded_line[6:]
                            if json_str == '[DONE]':
                                break
                            try:
                                json_data = json.loads(json_str)
                                if 'message' in json_data:
                                    yield json_data['message']
                            except json.JSONDecodeError:
                                pass