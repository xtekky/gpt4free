from __future__ import annotations

import asyncio
import json
import uuid
import cloudscraper
from typing import AsyncGenerator
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class Cloudflare(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://playground.ai.cloudflare.com"
    api_endpoint = "https://playground.ai.cloudflare.com/api/inference"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = '@cf/meta/llama-3.1-8b-instruct'
    models = [        
         '@cf/deepseek-ai/deepseek-math-7b-instruct', # Specific answer
         
         
         '@cf/thebloke/discolm-german-7b-v1-awq', 
         
         
         '@cf/tiiuae/falcon-7b-instruct', # Specific answer
         
         
         '@hf/google/gemma-7b-it', 


         '@cf/meta/llama-2-7b-chat-fp16', 
         '@cf/meta/llama-2-7b-chat-int8', 
         
         '@cf/meta/llama-3-8b-instruct', 
         '@cf/meta/llama-3-8b-instruct-awq', 
         default_model, 
         '@hf/meta-llama/meta-llama-3-8b-instruct', 
         
         '@cf/meta/llama-3.1-8b-instruct-awq', 
         '@cf/meta/llama-3.1-8b-instruct-fp8',  
         '@cf/meta/llama-3.2-11b-vision-instruct',  
         '@cf/meta/llama-3.2-1b-instruct',  
         '@cf/meta/llama-3.2-3b-instruct',  

         '@cf/mistral/mistral-7b-instruct-v0.1',
         '@hf/mistral/mistral-7b-instruct-v0.2',
         
         '@cf/openchat/openchat-3.5-0106',
         
         '@cf/microsoft/phi-2',
         
         '@cf/qwen/qwen1.5-0.5b-chat',
         '@cf/qwen/qwen1.5-1.8b-chat',
         '@cf/qwen/qwen1.5-14b-chat-awq',
         '@cf/qwen/qwen1.5-7b-chat-awq',
         
         '@cf/defog/sqlcoder-7b-2', # Specific answer
         
         '@cf/tinyllama/tinyllama-1.1b-chat-v1.0',
         
         '@cf/fblgit/una-cybertron-7b-v2-bf16',
    ]
    
    model_aliases = {       
        "german-7b-v1": "@cf/thebloke/discolm-german-7b-v1-awq",

        
        "gemma-7b": "@hf/google/gemma-7b-it",
        
        
        "llama-2-7b": "@cf/meta/llama-2-7b-chat-fp16",
        "llama-2-7b": "@cf/meta/llama-2-7b-chat-int8",
        
        "llama-3-8b": "@cf/meta/llama-3-8b-instruct",
        "llama-3-8b": "@cf/meta/llama-3-8b-instruct-awq",
        "llama-3-8b": "@cf/meta/llama-3.1-8b-instruct",
        "llama-3-8b": "@hf/meta-llama/meta-llama-3-8b-instruct",
        
        "llama-3.1-8b": "@cf/meta/llama-3.1-8b-instruct-awq",
        "llama-3.1-8b": "@cf/meta/llama-3.1-8b-instruct-fp8",
        "llama-3.1-8b": "@cf/meta/llama-3.1-8b-instruct-fp8",
        
        "llama-3.2-11b": "@cf/meta/llama-3.2-11b-vision-instruct",
        "llama-3.2-1b": "@cf/meta/llama-3.2-1b-instruct",
        "llama-3.2-3b": "@cf/meta/llama-3.2-3b-instruct",
        
        
        "mistral-7b": "@cf/mistral/mistral-7b-instruct-v0.1",
        "mistral-7b": "@hf/mistral/mistral-7b-instruct-v0.2",
        
        
        "openchat-3.5": "@cf/openchat/openchat-3.5-0106",
        
        
        "phi-2": "@cf/microsoft/phi-2",
        
                
        "qwen-1.5-0.5b": "@cf/qwen/qwen1.5-0.5b-chat",
        "qwen-1.5-1.8b": "@cf/qwen/qwen1.5-1.8b-chat",
        "qwen-1.5-14b": "@cf/qwen/qwen1.5-14b-chat-awq",
        "qwen-1.5-7b": "@cf/qwen/qwen1.5-7b-chat-awq",
        

        "tinyllama-1.1b": "@cf/tinyllama/tinyllama-1.1b-chat-v1.0",
        
        
        "cybertron-7b": "@cf/fblgit/una-cybertron-7b-v2-bf16",
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
        max_tokens: str = 2048,
        stream: bool = True,
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
        
        prompt = format_prompt(messages)
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ],
            "lora": None,
            "model": model,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        max_retries = 3
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
                            break
                        try:
                            content = json.loads(line[6:].decode('utf-8'))['response']
                            yield content
                        except Exception:
                            continue
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> str:
        full_response = ""
        async for response in cls.create_async_generator(model, messages, proxy, **kwargs):
            full_response += response
        return full_response
