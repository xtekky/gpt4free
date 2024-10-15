from __future__ import annotations

import json
import uuid
from aiohttp import ClientSession, ClientTimeout, ClientResponseError

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..image import ImageResponse

class AmigoChat(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://amigochat.io/chat/"
    chat_api_endpoint = "https://api.amigochat.io/v1/chat/completions"
    image_api_endpoint = "https://api.amigochat.io/v1/images/generations"
    working = True
    supports_gpt_4 = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'gpt-4o-mini'
    
    chat_models = [
        'gpt-4o',
        default_model,
        'o1-preview',
        'o1-mini',
        'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
        'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo',
        'claude-3-sonnet-20240229',
        'gemini-1.5-pro',
    ]
    
    image_models = [
        'flux-pro/v1.1',
        'flux-realism',
        'flux-pro',
        'dalle-e-3',
    ]
    
    models = [*chat_models, *image_models]
    
    model_aliases = {
        "o1": "o1-preview",
        "llama-3.1-405b": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "llama-3.2-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "claude-3.5-sonnet": "claude-3-sonnet-20240229",
        "gemini-pro": "gemini-1.5-pro",
        
        "flux-pro": "flux-pro/v1.1",
        "dalle-3": "dalle-e-3",
    }

    persona_ids = {
        'gpt-4o': "gpt",
        'gpt-4o-mini': "amigo",
        'o1-preview': "openai-o-one",
        'o1-mini': "openai-o-one-mini",
        'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo': "llama-three-point-one",
        'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo': "llama-3-2",
        'claude-3-sonnet-20240229': "claude",
        'gemini-1.5-pro': "gemini-1-5-pro",
        'flux-pro/v1.1': "flux-1-1-pro",
        'flux-realism': "flux-realism",
        'flux-pro': "flux-pro",
        'dalle-e-3': "dalle-three",
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases[model]
        else:
            return cls.default_chat_model if model in cls.chat_models else cls.default_image_model

    @classmethod
    def get_personaId(cls, model: str) -> str:
        return cls.persona_ids[model]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        stream: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        device_uuid = str(uuid.uuid4())
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                headers = {
                    "accept": "*/*",
                    "accept-language": "en-US,en;q=0.9",
                    "authorization": "Bearer",
                    "cache-control": "no-cache",
                    "content-type": "application/json",
                    "origin": cls.url,
                    "pragma": "no-cache",
                    "priority": "u=1, i",
                    "referer": f"{cls.url}/",
                    "sec-ch-ua": '"Chromium";v="129", "Not=A?Brand";v="8"',
                    "sec-ch-ua-mobile": "?0",
                    "sec-ch-ua-platform": '"Linux"',
                    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
                    "x-device-language": "en-US",
                    "x-device-platform": "web",
                    "x-device-uuid": device_uuid,
                    "x-device-version": "1.0.32"
                }
                
                async with ClientSession(headers=headers) as session:
                    if model in cls.chat_models:
                        # Chat completion
                        data = {
                            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                            "model": model,
                            "personaId": cls.get_personaId(model),
                            "frequency_penalty": 0,
                            "max_tokens": 4000,
                            "presence_penalty": 0,
                            "stream": stream,
                            "temperature": 0.5,
                            "top_p": 0.95
                        }
                        
                        timeout = ClientTimeout(total=300)  # 5 minutes timeout
                        async with session.post(cls.chat_api_endpoint, json=data, proxy=proxy, timeout=timeout) as response:
                            if response.status not in (200, 201):
                                error_text = await response.text()
                                raise Exception(f"Error {response.status}: {error_text}")
                            
                            async for line in response.content:
                                line = line.decode('utf-8').strip()
                                if line.startswith('data: '):
                                    if line == 'data: [DONE]':
                                        break
                                    try:
                                        chunk = json.loads(line[6:])  # Remove 'data: ' prefix
                                        if 'choices' in chunk and len(chunk['choices']) > 0:
                                            choice = chunk['choices'][0]
                                            if 'delta' in choice:
                                                content = choice['delta'].get('content')
                                            elif 'text' in choice:
                                                content = choice['text']
                                            else:
                                                content = None
                                            if content:
                                                yield content
                                    except json.JSONDecodeError:
                                        pass
                    else:
                        # Image generation
                        prompt = messages[0]['content']
                        data = {
                            "prompt": prompt,
                            "model": model,
                            "personaId": cls.get_personaId(model)
                        }
                        async with session.post(cls.image_api_endpoint, json=data, proxy=proxy) as response:
                            response.raise_for_status()
                            
                            response_data = await response.json()
                            
                            if "data" in response_data:
                                image_urls = []
                                for item in response_data["data"]:
                                    if "url" in item:
                                        image_url = item["url"]
                                        image_urls.append(image_url)
                                if image_urls:
                                    yield ImageResponse(image_urls, prompt)
                            else:
                                yield None
                
                break
            
            except (ClientResponseError, Exception) as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise e
                device_uuid = str(uuid.uuid4())
