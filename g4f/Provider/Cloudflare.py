from __future__ import annotations

from aiohttp import ClientSession
import asyncio
import json
import uuid
import cloudscraper
from typing import AsyncGenerator

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class Cloudflare(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Cloudflare AI"
    url = "https://playground.ai.cloudflare.com"
    api_endpoint = "https://playground.ai.cloudflare.com/api/inference"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = '@cf/meta/llama-3.1-8b-instruct-awq'
    models = [               
         '@cf/meta/llama-2-7b-chat-fp16', 
         '@cf/meta/llama-2-7b-chat-int8', 
         
         '@cf/meta/llama-3-8b-instruct', 
         '@cf/meta/llama-3-8b-instruct-awq', 
         '@hf/meta-llama/meta-llama-3-8b-instruct', 
         
         default_model, 
         '@cf/meta/llama-3.1-8b-instruct-fp8',  
         
         '@cf/meta/llama-3.2-1b-instruct',  

         '@hf/mistral/mistral-7b-instruct-v0.2',
         
         '@cf/qwen/qwen1.5-7b-chat-awq', 
         
         '@cf/defog/sqlcoder-7b-2',
    ]
    
    model_aliases = {       
        "llama-2-7b": "@cf/meta/llama-2-7b-chat-fp16",
        "llama-2-7b": "@cf/meta/llama-2-7b-chat-int8",
        
        "llama-3-8b": "@cf/meta/llama-3-8b-instruct",
        "llama-3-8b": "@cf/meta/llama-3-8b-instruct-awq",
        "llama-3-8b": "@hf/meta-llama/meta-llama-3-8b-instruct",
        
        "llama-3.1-8b": "@cf/meta/llama-3.1-8b-instruct-awq",
        "llama-3.1-8b": "@cf/meta/llama-3.1-8b-instruct-fp8",
        
        "llama-3.2-1b": "@cf/meta/llama-3.2-1b-instruct",
        
        "qwen-1.5-7b": "@cf/qwen/qwen1.5-7b-chat-awq",
        
        #"sqlcoder-7b": "@cf/defog/sqlcoder-7b-2",
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            return cls.default_model

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            'Accept': 'text/event-stream',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
            'Origin': cls.url,
            'Pragma': 'no-cache',
            'Referer': f'{cls.url}/',
            'Sec-Ch-Ua': '"Chromium";v="129", "Not=A?Brand";v="8"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Linux"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        }
        
        cookies = {
            '__cf_bm': uuid.uuid4().hex,
        }
        
        scraper = cloudscraper.create_scraper()
        
        data = {
            "messages": [
                {"role": "user", "content": format_prompt(messages)}
            ],
            "lora": None,
            "model": model,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        max_retries = 3
        full_response = ""
        
        for attempt in range(max_retries):
            try:
                response = scraper.post(
                    cls.api_endpoint,
                    headers=headers,
                    cookies=cookies,
                    json=data,
                    stream=True,
                    proxies={'http': proxy, 'https': proxy} if proxy else None
                )
                
                if response.status_code == 403:
                    await asyncio.sleep(2 ** attempt)
                    continue
                    
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line.startswith(b'data: '):
                        if line == b'data: [DONE]':
                            if full_response:
                                yield full_response
                            break
                        try:
                            content = json.loads(line[6:].decode('utf-8'))
                            if 'response' in content and content['response'] != '</s>':
                                yield content['response']
                        except Exception:
                            continue
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
