from __future__ import annotations

import json
import requests
from ..typing       import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..requests     import StreamSession

class DeepInfra(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://deepinfra.com"
    working = True
    supports_stream = True
    supports_message_history = True
    default_model = 'meta-llama/Llama-2-70b-chat-hf'
    
    @classmethod
    def get_models(cls):
        if not cls.models:
            url = 'https://api.deepinfra.com/models/featured'
            models = requests.get(url).json()
            cls.models = [model['model_name'] for model in models]
        return cls.models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool,
        proxy: str = None,
        timeout: int = 120,
        auth: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'Origin': 'https://deepinfra.com',
            'Referer': 'https://deepinfra.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'X-Deepinfra-Source': 'web-embed',
            'accept': 'text/event-stream',
            'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
        }
        if auth:
            headers['Authorization'] = f"bearer {auth}" 
            
        async with StreamSession(headers=headers,
            timeout=timeout,
            proxies={"https": proxy},
            impersonate="chrome110"
        ) as session:
            json_data = {
                'model'   : cls.get_model(model),
                'messages': messages,
                'temperature': kwargs.get("temperature", 0.7),
                'max_tokens': kwargs.get("max_tokens", 512),
                'stop': kwargs.get("stop", []),
                'stream'  : True
            }
            async with session.post('https://api.deepinfra.com/v1/openai/chat/completions',
                                    json=json_data) as response:
                response.raise_for_status()
                first = True
                async for line in response.iter_lines():
                    if not line.startswith(b"data: "):
                        continue
                    try:
                        json_line = json.loads(line[6:])
                        choices = json_line.get("choices", [{}])
                        finish_reason = choices[0].get("finish_reason")
                        if finish_reason:
                            break
                        token = choices[0].get("delta", {}).get("content")
                        if token:
                            if first:
                                token = token.lstrip()
                            if token:
                                first = False
                                yield token
                    except Exception:
                        raise RuntimeError(f"Response: {line}")
