from __future__ import annotations

import json
import random
import string

from aiohttp import ClientSession
from .. import debug

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

def generate_machine_id() :
    """
    generates random machine id
    Returns:
        str: machine id
    """    
    part1 = "".join(random.choices(string.digits, k=16))
    part2 = "".join(random.choices(string.digits + ".", k=25))
    return f"{part1}.{part2}"


class Chatai(AsyncGeneratorProvider, ProviderModelMixin):
    """
    Provider for Chatai
    """
    label = "Chatai"
    url = "https://chatai.aritek.app"  # Base URL
    api_endpoint = "https://chatai.aritek.app/stream" # API endpoint for chat
    working = True
    needs_auth = False 
    supports_stream = True
    supports_system_message = True 
    supports_message_history = True 

    default_model = 'gpt-4o-mini-2024-07-18'
    model_aliases = {"gpt-4o-mini":default_model} 
    models = list(model_aliases.keys())

    # --- ProviderModelMixin Methods ---
    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models or model == cls.default_model:
            return cls.default_model
        else:
            # Fallback to default if requested model is unknown
            return cls.default_model

    # --- AsyncGeneratorProvider Method ---
    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str | None = None,
        **kwargs
    ) -> AsyncResult:
        """
        Make an asynchronous request to the Chatai stream API.

        Args:
            model (str): The model name (currently ignored by this provider).
            messages (Messages): List of message dictionaries.
            proxy (str | None): Optional proxy URL.
            **kwargs: Additional arguments (currently unused).

        Yields:
            str: Chunks of the response text.

        Raises:
            Exception: If the API request fails.
        """
        
        # selected_model = cls.get_model(model) # Not sent in payload

        headers = {
            'Accept': 'text/event-stream',
            'Content-Type': 'application/json', 
            'User-Agent': 'Dalvik/2.1.0 (Linux; U; Android 7.1.2; SM-G935F Build/N2G48H)',
            'Host': 'chatai.aritek.app', 
            'Connection': 'Keep-Alive', 
        }

        static_machine_id = generate_machine_id()#"0343578260151264.464241743263788731"
        c_token = "eyJzdWIiOiIyMzQyZmczNHJ0MzR0MzQiLCJuYW1lIjoiSm9objM0NTM0NT"# might change 

        payload = {
            "machineId": static_machine_id,
            "msg": messages, # Pass the message list directly
            "token": c_token,
            "type": 0 
        }

        async with ClientSession(headers=headers) as session:
            try:
                async with session.post(
                    cls.api_endpoint,
                    json=payload,
                    proxy=proxy
                ) as response:
                    response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

                    # Process the Server-Sent Events (SSE) stream
                    async for line_bytes in response.content:
                        if not line_bytes:
                            continue # Skip empty linesw

                        line = line_bytes.decode('utf-8').strip()

                        if line.startswith("data:"):
                            data_str = line[len("data:"):].strip()

                            if data_str == "[DONE]":
                                break # End of stream signal

                            try:
                                chunk_data = json.loads(data_str)
                                choices = chunk_data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content_chunk = delta.get("content")
                                    if content_chunk:
                                        yield content_chunk
                                    # Check for finish reason if needed (e.g., to stop early)
                                    # finish_reason = choices[0].get("finish_reason")
                                    # if finish_reason:
                                    #     break
                            except json.JSONDecodeError:
                                debug.error(f"Warning: Could not decode JSON: {data_str}")
                                continue
                            except Exception as e:
                                debug.error(f"Warning: Error processing chunk: {e}")
                                continue
                        
            except Exception as e:
                # print()
                debug.error(f"Error during Chatai API request: {e}")
                raise e
