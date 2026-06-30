from __future__ import annotations

from typing import Any, Dict

from aiohttp import ClientTimeout

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt
from ..requests import StreamSession


class TeachAnything(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.teach-anything.com"
    api_endpoint = "/api/generate"
    
    working = True
    
    default_model = 'gemma'
    models = [default_model]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str | None = None,
        **kwargs: Any
    ) -> AsyncResult:
        headers = cls._get_headers()
        model = cls.get_model(model)
        
        async with StreamSession(headers=headers, impersonate="chrome") as session:
            prompt = format_prompt(messages)
            data = {"prompt": prompt}
            
            timeout = ClientTimeout(total=60)
            
            async with session.post(
                f"{cls.url}{cls.api_endpoint}",
                json=data,
                proxy=proxy,
                timeout=timeout
            ) as response:
                response.raise_for_status()
                buffer = b""
                async for chunk in response.iter_content():
                    buffer += chunk
                    try:
                        decoded = buffer.decode('utf-8')
                        yield decoded
                        buffer = b""
                    except UnicodeDecodeError:
                        # If we can't decode, we'll wait for more data
                        continue
                
                # Handle any remaining data in the buffer
                if buffer:
                    try:
                        yield buffer.decode('utf-8', errors='replace')
                    except Exception as e:
                        print(f"Error decoding final buffer: {e}")

    @staticmethod
    def _get_headers() -> Dict[str, str]:
        return {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": "https://www.teach-anything.com",
            "referer": "https://www.teach-anything.com/"
        }
