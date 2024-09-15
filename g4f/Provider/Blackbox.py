from __future__ import annotations

import re
import json
import random
import string
from aiohttp import ClientSession

from ..typing import AsyncResult, Messages, ImageType
from ..image import ImageResponse, to_data_uri
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

class Blackbox(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.blackbox.ai"
    api_endpoint = "https://www.blackbox.ai/api/chat"
    working = True
    supports_stream = True
    supports_system_message = True
    supports_message_history = True
    
    default_model = 'blackbox'
    models = [
        'blackbox',
        'gemini-1.5-flash',
        "llama-3.1-8b",
        'llama-3.1-70b',
        'llama-3.1-405b',
        'ImageGenerationLV45LJp'
    ]

    model_config = {
        "blackbox": {},
        "gemini-1.5-flash": {'mode': True, 'id': 'Gemini'},
        "llama-3.1-8b": {'mode': True, 'id': "llama-3.1-8b"},
        'llama-3.1-70b': {'mode': True, 'id': "llama-3.1-70b"},
        'llama-3.1-405b': {'mode': True, 'id': "llama-3.1-405b"},
        'ImageGenerationLV45LJp': {'mode': True, 'id': "ImageGenerationLV45LJp", 'name': "Image Generation"},
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
        image: ImageType = None,
        image_name: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": cls.url,
            "pragma": "no-cache",
            "referer": f"{cls.url}/",
            "sec-ch-ua": '"Not;A=Brand";v="24", "Chromium";v="128"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        }
        
        async with ClientSession(headers=headers) as session:
            if image is not None:
                messages[-1]["data"] = {
                    "fileText": image_name,
                    "imageBase64": to_data_uri(image)
                }
            
            random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=7))

            data = {
                "messages": messages,
                "id": random_id,
                "previewToken": None,
                "userId": None,
                "codeModelMode": True,
                "agentMode": {},
                "trendingAgentMode": {},
                "isMicMode": False,
                "maxTokens": None,
                "isChromeExt": False,
                "githubToken": None,
                "clickedAnswer2": False,
                "clickedAnswer3": False,
                "clickedForceWebSearch": False,
                "visitFromDelta": False,
                "mobileClient": False
            }

            if model == 'ImageGenerationLV45LJp':
                data["agentMode"] = cls.model_config[model]
            else:
                data["trendingAgentMode"] = cls.model_config[model]
            
            async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                if model == 'ImageGenerationLV45LJp':
                    response_text = await response.text()
                    url_match = re.search(r'https://storage\.googleapis\.com/[^\s\)]+', response_text)
                    if url_match:
                        image_url = url_match.group(0)
                        yield ImageResponse(image_url, alt=messages[-1]['content'])
                    else:
                        raise Exception("Image URL not found in the response")
                else:
                    async for chunk in response.content:
                        if chunk:
                            decoded_chunk = chunk.decode()
                            if decoded_chunk.startswith('$@$v=undefined-rv1$@$'):
                                decoded_chunk = decoded_chunk[len('$@$v=undefined-rv1$@$'):]
                            yield decoded_chunk
