from __future__ import annotations

import uuid
import secrets
import re
import base64
from aiohttp import ClientSession
from typing import AsyncGenerator, Optional

from ..typing import AsyncResult, Messages, ImageType
from ..image import to_data_uri, ImageResponse
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin

class Blackbox(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.blackbox.ai"
    working = True
    default_model = 'blackbox'
    models = [
        default_model,
        "gemini-1.5-flash",
        "llama-3.1-8b",
        'llama-3.1-70b',
        'llama-3.1-405b',
        'ImageGeneration',
    ]
    
    model_aliases = {
        "gemini-flash": "gemini-1.5-flash",
    }
    
    agent_mode_map = {
        'ImageGeneration': {"mode": True, "id": "ImageGenerationLV45LJp", "name": "Image Generation"},
    }

    model_id_map = {
        "blackbox": {},
        "gemini-1.5-flash": {'mode': True, 'id': 'Gemini'},
        "llama-3.1-8b": {'mode': True, 'id': "llama-3.1-8b"},
        'llama-3.1-70b': {'mode': True, 'id': "llama-3.1-70b"},
        'llama-3.1-405b': {'mode': True, 'id': "llama-3.1-405b"}
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
    async def download_image_to_base64_url(cls, url: str) -> str:
        async with ClientSession() as session:
            async with session.get(url) as response:
                image_data = await response.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
                mime_type = response.headers.get('Content-Type', 'image/jpeg')
                return f"data:{mime_type};base64,{base64_data}"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: Optional[str] = None,
        image: Optional[ImageType] = None,
        image_name: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[AsyncResult, None]:
        if image is not None:
            messages[-1]["data"] = {
                "fileText": image_name,
                "imageBase64": to_data_uri(image),
                "title": str(uuid.uuid4())
            }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": cls.url,
            "Content-Type": "application/json",
            "Origin": cls.url,
            "DNT": "1",
            "Sec-GPC": "1",
            "Alt-Used": "www.blackbox.ai",
            "Connection": "keep-alive",
        }

        async with ClientSession(headers=headers) as session:
            random_id = secrets.token_hex(16)
            random_user_id = str(uuid.uuid4())
            
            model = cls.get_model(model)  # Resolve the model alias
            
            data = {
                "messages": messages,
                "id": random_id,
                "userId": random_user_id,
                "codeModelMode": True,
                "agentMode": cls.agent_mode_map.get(model, {}),
                "trendingAgentMode": {},
                "isMicMode": False,
                "isChromeExt": False,
                "playgroundMode": False,
                "webSearchMode": False,
                "userSystemPrompt": "",
                "githubToken": None,
                "trendingAgentModel": cls.model_id_map.get(model, {}),
                "maxTokens": None
            }

            async with session.post(
                f"{cls.url}/api/chat", json=data, proxy=proxy
            ) as response:
                response.raise_for_status()
                full_response = ""
                buffer = ""
                image_base64_url = None
                async for chunk in response.content.iter_any():
                    if chunk:
                        decoded_chunk = chunk.decode()
                        cleaned_chunk = re.sub(r'\$@\$.+?\$@\$|\$@\$', '', decoded_chunk)
                        
                        buffer += cleaned_chunk
                        
                        # Check if there's a complete image line in the buffer
                        image_match = re.search(r'!\[Generated Image\]\((https?://[^\s\)]+)\)', buffer)
                        if image_match:
                            image_url = image_match.group(1)
                            # Download the image and convert to base64 URL
                            image_base64_url = await cls.download_image_to_base64_url(image_url)
                            
                            # Remove the image line from the buffer
                            buffer = re.sub(r'!\[Generated Image\]\(https?://[^\s\)]+\)', '', buffer)
                        
                        # Send text line by line
                        lines = buffer.split('\n')
                        for line in lines[:-1]:
                            if line.strip():
                                full_response += line + '\n'
                                yield line + '\n'
                        buffer = lines[-1]  # Keep the last incomplete line in the buffer

                # Send the remaining buffer if it's not empty
                if buffer.strip():
                    full_response += buffer
                    yield buffer

                # If an image was found, send it as ImageResponse
                if image_base64_url:
                    alt_text = "Generated Image"
                    image_response = ImageResponse(image_base64_url, alt=alt_text)
                    yield image_response
