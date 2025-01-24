from __future__ import annotations

import aiohttp
import json
import uuid
import re

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt
from ... import debug

class Qwen_Qwen_2_72B_Instruct(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://qwen-qwen2-72b-instruct.hf.space"
    api_endpoint = "https://qwen-qwen2-72b-instruct.hf.space/queue/join?"
    
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = False
    
    default_model = "qwen-qwen2-72b-instruct"
    models = [default_model]
    model_aliases = {"qwen-2-72b": default_model}

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        def generate_session_hash():
            """Generate a unique session hash."""
            return str(uuid.uuid4()).replace('-', '')[:12]

        # Generate a unique session hash
        session_hash = generate_session_hash()

        headers_join = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'origin': f'{cls.url}',
            'referer': f'{cls.url}/',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }

        # Prepare the prompt
        system_prompt = "\n".join([message["content"] for message in messages if message["role"] == "system"])
        messages = [message for message in messages if message["role"] != "system"]
        prompt = format_prompt(messages)

        payload_join = {
            "data": [prompt, [], system_prompt],
            "event_data": None,
            "fn_index": 0,
            "trigger_id": 11,
            "session_hash": session_hash
        }

        async with aiohttp.ClientSession() as session:
            # Send join request
            async with session.post(cls.api_endpoint, headers=headers_join, json=payload_join) as response:
                event_id = (await response.json())['event_id']

            # Prepare data stream request
            url_data = f'{cls.url}/queue/data'

            headers_data = {
                'accept': 'text/event-stream',
                'accept-language': 'en-US,en;q=0.9',
                'referer': f'{cls.url}/',
                'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
            }

            params_data = {
                'session_hash': session_hash
            }

            # Send data stream request
            async with session.get(url_data, headers=headers_data, params=params_data) as response:
                full_response = ""
                final_full_response = ""
                async for line in response.content:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        try:
                            json_data = json.loads(decoded_line[6:])

                            # Look for generation stages
                            if json_data.get('msg') == 'process_generating':
                                if 'output' in json_data and 'data' in json_data['output']:
                                    output_data = json_data['output']['data']
                                    if len(output_data) > 1 and len(output_data[1]) > 0:
                                        for item in output_data[1]:
                                            if isinstance(item, list) and len(item) > 1:
                                                fragment = str(item[1])
                                                # Ignore [0, 1] type fragments and duplicates
                                                if not re.match(r'^\[.*\]$', fragment) and not full_response.endswith(fragment):
                                                    full_response += fragment
                                                    yield fragment

                            # Check for completion
                            if json_data.get('msg') == 'process_completed':
                                # Final check to ensure we get the complete response
                                if 'output' in json_data and 'data' in json_data['output']:
                                    output_data = json_data['output']['data']
                                    if len(output_data) > 1 and len(output_data[1]) > 0:
                                        final_full_response = output_data[1][0][1]
                                        
                                        # Clean up the final response
                                        if final_full_response.startswith(full_response):
                                            final_full_response = final_full_response[len(full_response):]
                                        
                                        # Yield the remaining part of the final response
                                        if final_full_response:
                                            yield final_full_response
                                break

                        except json.JSONDecodeError:
                            debug.log("Could not parse JSON:", decoded_line)
