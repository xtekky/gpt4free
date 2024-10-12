from __future__ import annotations
import random
import json
import re
from aiohttp import ClientSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import ImageResponse

def split_long_message(message: str, max_length: int = 4000) -> list[str]:
    return [message[i:i+max_length] for i in range(0, len(message), max_length)]

class Airforce(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://api.airforce"
    image_api_endpoint = "https://api.airforce/imagine2"
    text_api_endpoint = "https://api.airforce/chat/completions"
    working = True
    
    default_model = 'llama-3-70b-chat'
    
    supports_gpt_35_turbo = True
    supports_gpt_4 = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    text_models = [
        'claude-3-haiku-20240307', 
        'claude-3-sonnet-20240229', 
        'claude-3-5-sonnet-20240620', 
        'claude-3-opus-20240229', 
        'chatgpt-4o-latest', 
        'gpt-4', 
        'gpt-4-turbo', 
        'gpt-4o-mini-2024-07-18', 
        'gpt-4o-mini', 
        'gpt-3.5-turbo', 
        'gpt-3.5-turbo-0125', 
        'gpt-3.5-turbo-1106', 
        default_model,
        'llama-3-70b-chat-turbo', 
        'llama-3-8b-chat', 
        'llama-3-8b-chat-turbo', 
        'llama-3-70b-chat-lite', 
        'llama-3-8b-chat-lite', 
        'llama-2-13b-chat', 
        'llama-3.1-405b-turbo', 
        'llama-3.1-70b-turbo', 
        'llama-3.1-8b-turbo', 
        'LlamaGuard-2-8b', 
        'Llama-Guard-7b', 
        'Llama-3.2-90B-Vision-Instruct-Turbo',
        'Mixtral-8x7B-Instruct-v0.1', 
        'Mixtral-8x22B-Instruct-v0.1', 
        'Mistral-7B-Instruct-v0.1', 
        'Mistral-7B-Instruct-v0.2', 
        'Mistral-7B-Instruct-v0.3', 
        'Qwen1.5-7B-Chat', 
        'Qwen1.5-14B-Chat', 
        'Qwen1.5-72B-Chat', 
        'Qwen1.5-110B-Chat', 
        'Qwen2-72B-Instruct', 
        'gemma-2b-it', 
        'gemma-2-9b-it', 
        'gemma-2-27b-it', 
        'gemini-1.5-flash', 
        'gemini-1.5-pro', 
        'deepseek-llm-67b-chat', 
        'Nous-Hermes-2-Mixtral-8x7B-DPO', 
        'Nous-Hermes-2-Yi-34B', 
        'WizardLM-2-8x22B', 
        'SOLAR-10.7B-Instruct-v1.0', 
        'MythoMax-L2-13b', 
        'cosmosrp', 
    ]
    
    image_models = [
        'flux',
        'flux-realism',
        'flux-anime',
        'flux-3d',
        'flux-disney',
        'flux-pixel',
        'flux-4o',
        'any-dark',
        'dall-e-3',
    ]
    
    models = [
        *text_models,
        *image_models,
    ]
    
    model_aliases = {
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "gpt-4o": "chatgpt-4o-latest",
        "llama-3-70b": "llama-3-70b-chat",
        "llama-3-8b": "llama-3-8b-chat",
        "mixtral-8x7b": "Mixtral-8x7B-Instruct-v0.1",
        "qwen-1.5-7b": "Qwen1.5-7B-Chat",
        "gemma-2b": "gemma-2b-it",
        "gemini-flash": "gemini-1.5-flash",
        "mythomax-l2-13b": "MythoMax-L2-13b",
        "solar-10.7b": "SOLAR-10.7B-Instruct-v1.0",
    }

    @classmethod
    def get_model(cls, model: str) -> str:
        if model in cls.models:
            return model
        elif model in cls.model_aliases:
            return cls.model_aliases.get(model, cls.default_model)
        else:
            return cls.default_model

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        seed: int = None,
        size: str = "1:1",
        stream: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)

        if model in cls.image_models:
            async for result in cls._generate_image(model, messages, proxy, seed, size):
                yield result
        elif model in cls.text_models:
            async for result in cls._generate_text(model, messages, proxy, stream):
                yield result
    
    @classmethod
    async def _generate_image(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        seed: int = None,
        size: str = "1:1",
        **kwargs
    ) -> AsyncResult:
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "origin": "https://llmplayground.net",
            "user-agent": "Mozilla/5.0"
        }

        if seed is None:
            seed = random.randint(0, 100000)

        prompt = messages[0]['content']

        async with ClientSession(headers=headers) as session:
            params = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "seed": seed
            }
            async with session.get(f"{cls.image_api_endpoint}", params=params, proxy=proxy) as response:
                response.raise_for_status()
                content_type = response.headers.get('Content-Type', '').lower()

                if 'application/json' in content_type:
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            yield chunk.decode('utf-8')
                elif 'image' in content_type:
                    image_data = b""
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            image_data += chunk
                    image_url = f"{cls.image_api_endpoint}?model={model}&prompt={prompt}&size={size}&seed={seed}"
                    alt_text = f"Generated image for prompt: {prompt}"
                    yield ImageResponse(images=image_url, alt=alt_text)

    @classmethod
    async def _generate_text(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        stream: bool = False,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "authorization": "Bearer missing api key",
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0"
        }

        async with ClientSession(headers=headers) as session:
            formatted_prompt = cls._format_messages(messages)
            prompt_parts = split_long_message(formatted_prompt)
            full_response = ""

            for part in prompt_parts:
                data = {
                    "messages": [{"role": "user", "content": part}],
                    "model": model,
                    "max_tokens": 4096,
                    "temperature": 1,
                    "top_p": 1,
                    "stream": stream
                }
                async with session.post(cls.text_api_endpoint, json=data, proxy=proxy) as response:
                    response.raise_for_status()
                    part_response = ""
                    if stream:
                        async for line in response.content:
                            if line:
                                line = line.decode('utf-8').strip()
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    json_data = json.loads(line[6:])
                                    content = json_data['choices'][0]['delta'].get('content', '')
                                    part_response += content
                    else:
                        json_data = await response.json()
                        content = json_data['choices'][0]['message']['content']
                        part_response = content

                    # Видаляємо повідомлення про перевищення ліміту символів
                    part_response = re.sub(
                        r"One message exceeds the \d+chars per message limit\..+https:\/\/discord\.com\/invite\/\S+",
                        '',
                        part_response
                    )
                    
                    part_response = re.sub(
                        r"Rate limit \(\d+\/minute\) exceeded\. Join our discord for more: .+https:\/\/discord\.com\/invite\/\S+",
                        '',
                        part_response
                    )

                    full_response += part_response
            yield full_response

    @classmethod
    def _format_messages(cls, messages: Messages) -> str:
        return " ".join([msg['content'] for msg in messages])
