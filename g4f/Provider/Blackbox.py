from __future__ import annotations

import uuid
import secrets
import re
from aiohttp import ClientSession, ClientResponse
from typing import AsyncGenerator, Optional

from ..typing import AsyncResult, Messages, ImageType
from ..image import to_data_uri
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
    ]
    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: Optional[str] = None,
        image: Optional[ImageType] = None,
        image_name: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
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
            model_id_map = {
                "blackbox": {},
                "gemini-1.5-flash": {'mode': True, 'id': 'Gemini'},
                "llama-3.1-8b": {'mode': True, 'id': "llama-3.1-8b"},
                'llama-3.1-70b': {'mode': True, 'id': "llama-3.1-70b"},
                'llama-3.1-405b': {'mode': True, 'id': "llama-3.1-405b"}
            }
            data = {
                "messages": messages,
                "id": random_id,
                "userId": random_user_id,
                "codeModelMode": True,
                "agentMode": {},
                "trendingAgentMode": {},
                "isMicMode": False,
                "isChromeExt": False,
                "playgroundMode": False,
                "webSearchMode": False,
                "userSystemPrompt": "",
                "githubToken": None,
                "trendingAgentModel": model_id_map[model], # if you actually test this on the site, just ask each model "yo", weird behavior imo
                "maxTokens": None
            }

            async with session.post(
                f"{cls.url}/api/chat", json=data, proxy=proxy
            ) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_any():
                    if chunk:
                        # Decode the chunk and clean up unwanted prefixes using a regex
                        decoded_chunk = chunk.decode()
                        cleaned_chunk = re.sub(r'\$@\$.+?\$@\$|\$@\$', '', decoded_chunk)
                        yield cleaned_chunk
