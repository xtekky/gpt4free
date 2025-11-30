from __future__ import annotations

import aiohttp
import json
import uuid

from ...typing import AsyncResult, Messages
from ...providers.response import JsonConversation
from ...requests.raise_for_status import raise_for_status
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt, get_last_user_message, get_system_prompt
from ... import debug

class BAAI_Ling(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Ling & Ring Playground"
    url = "https://cafe3310-ling-playground.hf.space"
    api_endpoint = f"{url}/gradio_api/queue/join"

    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = False

    default_model = "ling-1t"
    model_aliases = {
        "ling": default_model,
    }
    models = ['ling-mini-2.0', 'ling-1t', 'ling-flash-2.0', 'ring-1t', 'ring-flash-2.0', 'ring-mini-2.0']

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        conversation: JsonConversation = None,
        **kwargs
    ) -> AsyncResult:
        is_new_conversation = conversation is None or not hasattr(conversation, 'session_hash')
        if is_new_conversation:
            conversation = JsonConversation(session_hash=str(uuid.uuid4()).replace('-', '')[:12])

        model = cls.get_model(model)
        prompt = format_prompt(messages) if is_new_conversation else get_last_user_message(messages)

        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'origin': cls.url,
            'referer': f'{cls.url}/',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
        }

        payload = {
            "data": [
                prompt,
                [
                    [
                        None,
                        "Hello! I'm Ling. Try selecting a scenario and a message example below to get started."
                    ]
                ],
                get_system_prompt(messages),
                1,
                model
            ],
            "event_data": None,
            "fn_index": 11,
            "trigger_id": 14,
            "session_hash": conversation.session_hash
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(cls.api_endpoint, headers=headers, json=payload, proxy=proxy) as response:
                await raise_for_status(response)
                # Response body must be consumed for the request to complete
                await response.json()

            data_url = f'{cls.url}/gradio_api/queue/data?session_hash={conversation.session_hash}'
            headers_data = {
                'accept': 'text/event-stream',
                'referer': f'{cls.url}/',
                'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
            }

            async with session.get(data_url, headers=headers_data, proxy=proxy) as response:
                full_response = ""
                async for line in response.content:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        try:
                            json_data = json.loads(decoded_line[6:])
                            if json_data.get('msg') == 'process_generating':
                                if 'output' in json_data and 'data' in json_data['output']:
                                    output_data = json_data['output']['data']
                                    if output_data and len(output_data) > 0:
                                        parts = output_data[0][0]
                                        if len(parts) == 2:
                                            new_text = output_data[0][1].pop()
                                            full_response += new_text
                                            yield new_text
                                        if len(parts) > 2:
                                            new_text = parts[2]
                                            full_response += new_text
                                            yield new_text

                            elif json_data.get('msg') == 'process_completed':
                               break

                        except json.JSONDecodeError:
                            debug.log("Could not parse JSON:", decoded_line)
