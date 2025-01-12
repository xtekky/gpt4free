from __future__ import annotations

import uuid
import json

from aiohttp import ClientSession

from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

class ChatGLM(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://chatglm.cn"
    api_endpoint = "https://chatglm.cn/chatglm/mainchat-api/guest/stream"
    
    working = True
    supports_stream = True
    supports_system_message = False
    supports_message_history = False
    
    default_model = "all-tools-230b"
    models = [default_model]
    model_aliases = {"glm-4": default_model}

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        device_id = str(uuid.uuid4()).replace('-', '')
        
        headers = {
            'Accept-Language': 'en-US,en;q=0.9',
            'App-Name': 'chatglm',
            'Authorization': 'undefined',
            'Content-Type': 'application/json',
            'Origin': 'https://chatglm.cn',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'X-App-Platform': 'pc',
            'X-App-Version': '0.0.1',
            'X-Device-Id': device_id,
            'Accept': 'text/event-stream'
        }
        
        async with ClientSession(headers=headers) as session:
            data = {
                "assistant_id": "65940acff94777010aa6b796",
                "conversation_id": "",
                "meta_data": {
                    "if_plus_model": False,
                    "is_test": False,
                    "input_question_type": "xxxx",
                    "channel": "",
                    "draft_id": "",
                    "quote_log_id": "",
                    "platform": "pc"
                },
                "messages": [
                    {
                        "role": message["role"],
                        "content": [
                            {
                                "type": "text",
                                "text": message["content"]
                            }
                        ]
                    }
                    for message in messages
                ]
            }
            
            yield_text = 0
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                await raise_for_status(response)
                async for chunk in response.content:
                    if chunk:
                        decoded_chunk = chunk.decode('utf-8')
                        if decoded_chunk.startswith('data: '):
                            try:
                                json_data = json.loads(decoded_chunk[6:])
                                parts = json_data.get('parts', [])
                                if parts:
                                    content = parts[0].get('content', [])
                                    if content:
                                        text = content[0].get('text', '')[yield_text:]
                                        if text:
                                            yield text
                                            yield_text += len(text)
                            except json.JSONDecodeError:
                                pass
