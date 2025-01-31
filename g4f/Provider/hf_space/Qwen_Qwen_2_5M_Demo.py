from __future__ import annotations

import aiohttp
import json
import uuid

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt
from ...providers.response import JsonConversation, Reasoning
from ..helper import get_last_user_message
from ... import debug

class Qwen_Qwen_2_5M_Demo(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://qwen-qwen2-5-1m-demo.hf.space"
    api_endpoint = f"{url}/run/predict?__theme=light"

    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = False

    default_model = "qwen-2.5-1m-demo"
    models = [default_model]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        return_conversation: bool = False,
        conversation: JsonConversation = None,
        **kwargs
    ) -> AsyncResult:
        def generate_session_hash():
            """Generate a unique session hash."""
            return str(uuid.uuid4()).replace('-', '')[:12]

        # Generate a unique session hash
        session_hash = generate_session_hash() if conversation is None else getattr(conversation, "session_hash")
        if return_conversation:
            yield JsonConversation(session_hash=session_hash)

        prompt = format_prompt(messages) if conversation is None else get_last_user_message(messages)

        headers = {
            'accept': '*/*',
            'accept-language': 'en-US',
            'content-type': 'application/json',
            'origin': cls.url,
            'referer': f'{cls.url}/?__theme=light',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
        }

        payload_predict = {
            "data":[{"files":[],"text":prompt},[],[]],
            "event_data": None,
            "fn_index": 1,
            "trigger_id": 5,
            "session_hash": session_hash
        }

        async with aiohttp.ClientSession() as session:
            # Send join request
            async with session.post(cls.api_endpoint, headers=headers, json=payload_predict) as response:
                data = (await response.json())['data']

            join_url = f"{cls.url}/queue/join?__theme=light"
            join_data = {"data":[[[{"id":None,"elem_id":None,"elem_classes":None,"name":None,"text":prompt,"flushing":None,"avatar":"","files":[]},None]],None,0],"event_data":None,"fn_index":2,"trigger_id":5,"session_hash":session_hash}

            async with session.post(join_url, headers=headers, json=join_data) as response:
                event_id = (await response.json())['event_id']

            # Prepare data stream request
            url_data = f'{cls.url}/queue/data?session_hash={session_hash}'

            headers_data = {
                'accept': 'text/event-stream',
                'referer': f'{cls.url}/?__theme=light',
                'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
            }
            # Send data stream request
            async with session.get(url_data, headers=headers_data) as response:
                yield_response = ""
                yield_response_len = 0
                async for line in response.content:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        try:
                            json_data = json.loads(decoded_line[6:])

                            # Look for generation stages
                            if json_data.get('msg') == 'process_generating':
                                if 'output' in json_data and 'data' in json_data['output'] and json_data['output']['data'][0]:
                                    output_data = json_data['output']['data'][0][0]
                                    if len(output_data) > 2:
                                        text = output_data[2].split("\n<summary>")[0]
                                        if text == "Qwen is thinking...":
                                            yield Reasoning(None, text)
                                        elif text.startswith(yield_response):
                                            yield text[yield_response_len:]
                                        else:
                                            yield text
                                        yield_response_len = len(text)
                                        yield_response = text

                            # Check for completion
                            if json_data.get('msg') == 'process_completed':
                                # Final check to ensure we get the complete response
                                if 'output' in json_data and 'data' in json_data['output']:
                                    output_data = json_data['output']['data'][0][0][1][0]["text"].split("\n<summary>")[0]
                                    yield output_data[yield_response_len:]
                                    yield_response_len = len(text)
                                break

                        except json.JSONDecodeError:
                            debug.log("Could not parse JSON:", decoded_line)
