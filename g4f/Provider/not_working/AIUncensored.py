from __future__ import annotations

from aiohttp import ClientSession
import time
import hmac
import hashlib
import json
import random

from ...typing import AsyncResult, Messages
from ...requests.raise_for_status import raise_for_status
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..helper import format_prompt
from ...providers.response import FinishReason

class AIUncensored(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.aiuncensored.info/ai_uncensored"
    api_key = "62852b00cb9e44bca86f0ec7e7455dc6"
    
    working = False
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = "hermes3-70b"
    models = [default_model]
    
    model_aliases = {"hermes-3": "hermes3-70b"}

    @staticmethod
    def calculate_signature(timestamp: str, json_dict: dict) -> str:
        message = f"{timestamp}{json.dumps(json_dict)}"
        secret_key = b'your-super-secret-key-replace-in-production'
        signature = hmac.new(
            secret_key,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    @staticmethod
    def get_server_url() -> str:
        servers = [
            "https://llm-server-nov24-ibak.onrender.com",
            "https://llm-server-nov24-qv2w.onrender.com", 
            "https://llm-server-nov24.onrender.com"
        ]
        return random.choice(servers)

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = False,
        proxy: str = None,
        api_key: str = None,
        **kwargs
    ) -> AsyncResult:      
        model = cls.get_model(model)
        
        timestamp = str(int(time.time()))
        
        json_dict = {
            "messages": [{"role": "user", "content": format_prompt(messages)}],
            "model": model,
            "stream": stream
        }
        
        signature = cls.calculate_signature(timestamp, json_dict)
        
        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'origin': 'https://www.aiuncensored.info',
            'referer': 'https://www.aiuncensored.info/',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'x-api-key': cls.api_key,
            'x-timestamp': timestamp,
            'x-signature': signature
        }
        
        url = f"{cls.get_server_url()}/api/chat"
        
        async with ClientSession(headers=headers) as session:
            async with session.post(url, json=json_dict, proxy=proxy) as response:
                await raise_for_status(response)
                
                if stream:
                    full_response = ""
                    async for line in response.content:
                        if line:
                            try:
                                line_text = line.decode('utf-8')
                                if line_text.startswith(''):
                                    data = line_text[6:]
                                    if data == '[DONE]':
                                        yield FinishReason("stop")
                                        break
                                    try:
                                        json_data = json.loads(data)
                                        if 'data' in json_data:
                                            yield json_data['data']
                                            full_response += json_data['data']
                                    except json.JSONDecodeError:
                                        continue
                            except UnicodeDecodeError:
                                continue
                    if full_response:
                        yield FinishReason("length")
                else:
                    response_json = await response.json()
                    if 'content' in response_json:
                        yield response_json['content']
                        yield FinishReason("length")
