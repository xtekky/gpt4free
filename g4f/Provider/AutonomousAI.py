from __future__ import annotations

from aiohttp import ClientSession
import base64
import json

from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class AutonomousAI(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.autonomous.ai/anon/"
    api_endpoints = {
        "llama": "https://chatgpt.autonomous.ai/api/v1/ai/chat",
        "qwen_coder": "https://chatgpt.autonomous.ai/api/v1/ai/chat",
        "hermes": "https://chatgpt.autonomous.ai/api/v1/ai/chat-hermes",
        "vision": "https://chatgpt.autonomous.ai/api/v1/ai/chat-vision",
        "summary": "https://chatgpt.autonomous.ai/api/v1/ai/summary"
    }
    
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "llama"
    models = [default_model, "qwen_coder", "hermes", "vision", "summary"]
    
    model_aliases = {
        "llama-3.3-70b": default_model,
        "qwen-2.5-coder-32b": "qwen_coder",
        "hermes-3": "hermes",
        "llama-3.2-90b": "vision",
        "llama-3.3-70b": "summary"
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        stream: bool = False,
        **kwargs
    ) -> AsyncResult:
        api_endpoint = cls.api_endpoints[model]
        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'country-code': 'US',
            'origin': 'https://www.autonomous.ai',
            'referer': 'https://www.autonomous.ai/',
            'time-zone': 'America/New_York',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }

        async with ClientSession(headers=headers) as session:
            prompt = format_prompt(messages)
            
            # Encode message
            message = [{"role": "user", "content": prompt}]
            message_json = json.dumps(message)
            encoded_message = base64.b64encode(message_json.encode('utf-8')).decode('utf-8')
            
            data = {
                "messages": encoded_message,
                "threadId": model,
                "stream": stream,
                "aiAgent": model
            }
            
            async with session.post(api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                async for chunk in response.content:
                    if chunk:
                        chunk_str = chunk.decode()
                        if chunk_str == "data: [DONE]":
                            continue
                        
                        try:
                            # Remove "data: " prefix and parse JSON
                            chunk_data = json.loads(chunk_str.replace("data: ", ""))
                            if "choices" in chunk_data and chunk_data["choices"]:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
