from __future__ import annotations

from typing import Any, Dict

from aiohttp import ClientSession, ClientTimeout

from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt


class TeachAnything(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.teach-anything.com"
    api_endpoint = "/api/generate"
    
    working = True
    
    default_model = 'gemini-1.5-pro'
    models = [default_model, 'gemini-1.5-flash']

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
        
        async with ClientSession(headers=headers) as session:
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
                async for chunk in response.content.iter_any():
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
            "dnt": "1",
            "origin": "https://www.teach-anything.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://www.teach-anything.com/",
            "sec-ch-us": '"Not?A_Brand";v="99", "Chromium";v="130"',
            "sec-ch-us-mobile": "?0",
            "sec-ch-us-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        }
