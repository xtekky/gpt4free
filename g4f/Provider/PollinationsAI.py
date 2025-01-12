from __future__ import annotations

import json
import random
import requests
from urllib.parse import quote
from typing import Optional
from aiohttp import ClientSession

from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..requests.raise_for_status import raise_for_status
from ..typing import AsyncResult, Messages
from ..image import ImageResponse

class PollinationsAI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "Pollinations AI"
    url = "https://pollinations.ai"
    
    working = True
    supports_stream = False
    supports_system_message = True
    supports_message_history = True

    # API endpoints base
    api_base = "https://text.pollinations.ai/openai"

    # API endpoints
    text_api_endpoint = "https://text.pollinations.ai/"
    image_api_endpoint = "https://image.pollinations.ai/"

    # Models configuration
    default_model = "openai"
    default_image_model = "flux"

    image_models = []
    models = []

    additional_models_image = ["midjourney", "dall-e-3"]
    additional_models_text = ["claude", "karma", "command-r", "llamalight", "mistral-large", "sur", "sur-mistral"]
    model_aliases = {
        "gpt-4o": default_model,
        "qwen-2-72b": "qwen",
        "qwen-2.5-coder-32b": "qwen-coder",
        "llama-3.3-70b": "llama",
        "mistral-nemo": "mistral",
        #"": "karma",
        "gpt-4": "searchgpt",
        "gpt-4": "claude",
        "claude-3.5-sonnet": "sur",
        "deepseek-chat": "deepseek",
        "llama-3.2-3b": "llamalight", 
    }

    @classmethod
    def get_models(cls, **kwargs):
        # Initialize model lists if not exists
        if not hasattr(cls, 'image_models'):
            cls.image_models = []
        if not hasattr(cls, 'text_models'):
            cls.text_models = []

        # Fetch image models if not cached
        if not cls.image_models:
            url = "https://image.pollinations.ai/models"
            response = requests.get(url)
            raise_for_status(response)
            cls.image_models = response.json()
            cls.image_models.extend(cls.additional_models_image)

        # Fetch text models if not cached
        if not cls.text_models:
            url = "https://text.pollinations.ai/models"
            response = requests.get(url)
            raise_for_status(response)
            cls.text_models = [model.get("name") for model in response.json()]
            cls.text_models.extend(cls.additional_models_text)

        # Return combined models
        return cls.text_models + cls.image_models

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        # Image specific parameters
        prompt: str = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        nologo: bool = True,
        private: bool = False,
        enhance: bool = False,
        safe: bool = False,
        # Text specific parameters
        temperature: float = 0.5,
        presence_penalty: float = 0,
        top_p: float = 1,
        frequency_penalty: float = 0,
        stream: bool = False,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)

        # Check if models
        # Image generation
        if model in cls.image_models:
            async for result in cls._generate_image(
                model=model,
                messages=messages,
                prompt=prompt,
                proxy=proxy,
                width=width,
                height=height,
                seed=seed,
                nologo=nologo,
                private=private,
                enhance=enhance,
                safe=safe
            ):
                yield result
        else:
            # Text generation
            async for result in cls._generate_text(
                model=model,
                messages=messages,
                proxy=proxy,
                temperature=temperature,
                presence_penalty=presence_penalty,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                stream=stream
            ):
                yield result

    @classmethod
    async def _generate_image(
        cls,
        model: str,
        messages: Messages,
        prompt: str,
        proxy: str,
        width: int,
        height: int,
        seed: Optional[int],
        nologo: bool,
        private: bool,
        enhance: bool,
        safe: bool
    ) -> AsyncResult:
        if seed is None:
            seed = random.randint(0, 10000)

        headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
        }

        params = {
            "seed": seed,
            "width": width,
            "height": height,
            "model": model,
            "nologo": nologo,
            "private": private,
            "enhance": enhance,
            "safe": safe
        }
        params = {k: v for k, v in params.items() if v is not None}

        async with ClientSession(headers=headers) as session:
            prompt = messages[-1]["content"] if prompt is None else prompt
            param_string = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{cls.image_api_endpoint}/prompt/{quote(prompt)}?{param_string}"

            async with session.head(url, proxy=proxy) as response:
                if response.status == 200:
                    image_response = ImageResponse(images=url, alt=prompt)
                    yield image_response

    @classmethod
    async def _generate_text(
        cls,
        model: str,
        messages: Messages,
        proxy: str,
        temperature: float,
        presence_penalty: float,
        top_p: float,
        frequency_penalty: float,
        stream: bool,
        seed: Optional[int] = None
    ) -> AsyncResult:       
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }
        
        if seed is None:
            seed = random.randint(0, 10000)
			
        async with ClientSession(headers=headers) as session:
            data = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "presence_penalty": presence_penalty,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "jsonMode": False,
                "stream": stream,
                "seed": seed,
                "cache": False
            }

            async with session.post(cls.text_api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                async for chunk in response.content:
                    if chunk:
                        decoded_chunk = chunk.decode()
                        
                        # Skip [DONE].
                        if "data: [DONE]" in decoded_chunk:
                            continue
                            
                        # Processing plain text
                        if not decoded_chunk.startswith("data:"):
                            clean_text = decoded_chunk.strip()
                            if clean_text:
                                yield clean_text
                            continue
                        
                        # Processing JSON format
                        try:
                            # Remove the prefix “data: “ and parse JSON
                            json_str = decoded_chunk.replace("data:", "").strip()
                            json_response = json.loads(json_str)
                            
                            if "choices" in json_response and json_response["choices"]:
                                if "delta" in json_response["choices"][0]:
                                    content = json_response["choices"][0]["delta"].get("content")
                                    if content:
                                        # Remove escaped slashes before parentheses
                                        clean_content = content.replace("\\(", "(").replace("\\)", ")")
                                        yield clean_content
                        except json.JSONDecodeError:
                            # If JSON could not be parsed, skip
                            continue
