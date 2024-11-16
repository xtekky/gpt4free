from __future__ import annotations

import random
import json
import re

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import ImageResponse
from ..requests import StreamSession, raise_for_status
from .airforce.AirforceChat import AirforceChat
from .airforce.AirforceImage import AirforceImage

class Airforce(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://api.airforce"
    api_endpoint_completions = AirforceChat.api_endpoint
    api_endpoint_imagine = AirforceImage.api_endpoint
    working = True
    default_model = "gpt-4o-mini"
    supports_system_message = True
    supports_message_history = True
    text_models = [
        'gpt-4-turbo',
        default_model,
        'llama-3.1-70b-turbo',
        'llama-3.1-8b-turbo',
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
    ]
    models = [
        *text_models,
        *image_models,
    ]
    model_aliases = {
        "gpt-4o": "chatgpt-4o-latest",
        "llama-3.1-70b": "llama-3.1-70b-turbo",
        "llama-3.1-8b": "llama-3.1-8b-turbo",
        "gpt-4": "gpt-4-turbo",
    }

    @classmethod
    def create_async_generator(
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
            return cls._generate_image(model, messages, proxy, seed, size)
        else:
            return cls._generate_text(model, messages, proxy, stream, **kwargs)

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
        prompt = messages[-1]['content']

        async with StreamSession(headers=headers, proxy=proxy) as session:
            params = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "seed": seed
            }
            async with session.get(f"{cls.api_endpoint_imagine}", params=params) as response:
                await raise_for_status(response)
                content_type = response.headers.get('Content-Type', '').lower()

                if 'application/json' in content_type:
                    raise RuntimeError(await response.json().get("error", {}).get("message"))
                elif 'image' in content_type:
                    image_data = b""
                    async for chunk in response.iter_content():
                        if chunk:
                            image_data += chunk
                    image_url = f"{cls.api_endpoint_imagine}?model={model}&prompt={prompt}&size={size}&seed={seed}"
                    yield ImageResponse(images=image_url, alt=prompt)

    @classmethod
    async def _generate_text(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        stream: bool = False,
        max_tokens: int = 4096,
        temperature: float = 1,
        top_p: float = 1,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "authorization": "Bearer missing api key",
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0"
        }
        async with StreamSession(headers=headers, proxy=proxy) as session:
            data = {
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream
            }
            async with session.post(cls.api_endpoint_completions, json=data) as response:
                await raise_for_status(response)
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/json' in content_type:
                    json_data = await response.json()
                    if json_data.get("model") == "error":
                        raise RuntimeError(json_data['choices'][0]['message'].get('content', ''))
                if stream:
                    async for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8').strip()
                            if line.startswith("data: ") and line != "data: [DONE]":
                                json_data = json.loads(line[6:])
                                content = json_data['choices'][0]['delta'].get('content', '')
                                if content:
                                    yield cls._filter_content(content)
                else:
                    json_data = await response.json()
                    content = json_data['choices'][0]['message']['content']
                    yield cls._filter_content(content)

    @classmethod
    def _filter_content(cls, part_response: str) -> str:
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
        return part_response