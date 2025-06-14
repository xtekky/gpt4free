from __future__ import annotations

import aiohttp
import json
import uuid

from ...typing import AsyncResult, Messages
from ...providers.response import Reasoning, JsonConversation
from ...requests.raise_for_status import raise_for_status
from ...errors import ModelNotFoundError
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import get_last_user_message, get_system_prompt
from ... import debug

class Qwen_Qwen_3(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Qwen Qwen-3"
    url = "https://qwen-qwen3-demo.hf.space"
    api_endpoint = "https://qwen-qwen3-demo.hf.space/gradio_api/queue/join?__theme=system"

    working = True
    supports_stream = True
    supports_system_message = True

    default_model = "qwen-3-235b"
    models = {
        default_model,
        "qwen-3-32b",
        "qwen-3-30b-a3b",
        "qwen-3-14b",
        "qwen-3-8b",
        "qwen-3-4b",
        "qwen-3-1.7b",
        "qwen-3-0.6b",
    }
    model_aliases = {
        "qwen-3-235b": "qwen3-235b-a22b",
        "qwen-3-30b": "qwen3-30b-a3b",
        "qwen-3-32b": "qwen3-32b",
        "qwen-3-14b": "qwen3-14b",
        "qwen-3-4b": "qwen3-4b",
        "qwen-3-1.7b": "qwen3-1.7b",
        "qwen-3-0.6b": "qwen3-0.6b"
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        conversation: JsonConversation = None,
        thinking_budget: int = 38,
        **kwargs
    ) -> AsyncResult:
        try:
            model = cls.get_model(model)
        except ModelNotFoundError:
            pass
        if conversation is None or not hasattr(conversation, 'session_hash'):
            conversation = JsonConversation(session_hash=str(uuid.uuid4()).replace('-', ''))

        headers_join = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Referer': f'{cls.url}/?__theme=system',
            'content-type': 'application/json',
            'Origin': cls.url,
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
        }

        system_prompt = get_system_prompt(messages)
        system_prompt = system_prompt if system_prompt else "You are a helpful and harmless assistant."

        payload_join = {"data": [
            get_last_user_message(messages),
            {"thinking_budget": thinking_budget, "model": cls.get_model(model), "sys_prompt": system_prompt}, None, None],
            "event_data": None, "fn_index": 13, "trigger_id": 31, "session_hash": conversation.session_hash
        }

        async with aiohttp.ClientSession() as session:
            # Send join request
            async with session.post(cls.api_endpoint, headers=headers_join, json=payload_join, proxy=proxy) as response:
                await raise_for_status(response)
                (await response.json())['event_id']

            # Prepare data stream request
            url_data = f'{cls.url}/gradio_api/queue/data'

            headers_data = {
                'Accept': 'text/event-stream',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': f'{cls.url}/?__theme=system',
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0',
            }

            params_data = {
                'session_hash': conversation.session_hash,
            }

            # Send data stream request
            async with session.get(url_data, headers=headers_data, params=params_data, proxy=proxy) as response:
                is_thinking = False
                async for line in response.content:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        try:
                            json_data = json.loads(decoded_line[6:])

                            # Look for generation stages
                            if json_data.get('msg') == 'process_generating':
                                if 'output' in json_data and 'data' in json_data['output'] and len(
                                        json_data['output']['data']) > 5:
                                    updates = json_data['output']['data'][5]
                                    for update in updates:
                                        if isinstance(update[2], dict):
                                            if update[2].get('type') == 'tool':
                                                yield Reasoning(update[2].get('content'),
                                                                status=update[2].get('options', {}).get('title'))
                                                is_thinking = True
                                            elif update[2].get('type') == 'text':
                                                yield update[2].get('content')
                                                is_thinking = False
                                        elif isinstance(update, list) and isinstance(update[1], list) and len(
                                                update[1]) > 4:
                                            if update[1][4] == "content":
                                                yield Reasoning(update[2]) if is_thinking else update[2]
                                            elif update[1][4] == "options":
                                                if update[2] != "done":
                                                    yield Reasoning(status=update[2])
                                                is_thinking = False
                            # Check for completion
                            if json_data.get('msg') == 'process_completed':
                                break

                        except json.JSONDecodeError:
                            debug.log("Could not parse JSON:", decoded_line)
