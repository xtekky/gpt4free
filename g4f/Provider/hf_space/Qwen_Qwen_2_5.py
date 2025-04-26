from __future__ import annotations

import aiohttp
import json
import uuid
import re

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt
from ... import debug

class Qwen_Qwen_2_5(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Qwen Qwen-2.5"
    url = "https://qwen-qwen2-5.hf.space"
    api_endpoint = "https://qwen-qwen2-5.hf.space/queue/join"
    
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = False
    
    default_model = "qwen-qwen2-5"
    model_aliases = {"qwen-2.5": default_model}
    models = list(model_aliases.keys())

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
            return str(uuid.uuid4()).replace('-', '')[:10]

        # Generate a unique session hash
        session_hash = generate_session_hash()

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

        # Prepare the prompt
        system_prompt = "\n".join([message["content"] for message in messages if message["role"] == "system"])
        if not system_prompt:
            system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        messages = [message for message in messages if message["role"] != "system"]
        prompt = format_prompt(messages)

        payload_join = {
            "data": [prompt, [], system_prompt, "72B"],
            "event_data": None,
            "fn_index": 3,
            "trigger_id": 25,
            "session_hash": session_hash
        }

        async with aiohttp.ClientSession() as session:
            # Send join request
            async with session.post(cls.api_endpoint, headers=headers_join, json=payload_join) as response:
                event_id = (await response.json())['event_id']

            # Prepare data stream request
            url_data = f'{cls.url}/queue/data'

            headers_data = {
                'Accept': 'text/event-stream',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': f'{cls.url}/?__theme=system',
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0',
            }

            params_data = {
                'session_hash': session_hash
            }

            # Send data stream request
            async with session.get(url_data, headers=headers_data, params=params_data) as response:
                full_response = ""
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
                                                # Extract the fragment, handling both string and dict types
                                                fragment = item[1]
                                                if isinstance(fragment, dict) and 'text' in fragment:
                                                    # For the first chunk, extract only the text part
                                                    fragment = fragment['text']
                                                else:
                                                    fragment = str(fragment)
                                                
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
                                        # Get the final response text
                                        response_item = output_data[1][0][1]
                                        if isinstance(response_item, dict) and 'text' in response_item:
                                            final_full_response = response_item['text']
                                        else:
                                            final_full_response = str(response_item)
                                        
                                        # Clean up the final response
                                        if isinstance(final_full_response, str) and final_full_response.startswith(full_response):
                                            final_text = final_full_response[len(full_response):]
                                        else:
                                            final_text = final_full_response
                                        
                                        # Yield the remaining part of the final response
                                        if final_text and final_text != full_response:
                                            yield final_text
                                break

                        except json.JSONDecodeError:
                            debug.log("Could not parse JSON:", decoded_line)
