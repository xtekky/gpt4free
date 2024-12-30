from __future__ import annotations

import random
import asyncio
import re
import json
from pathlib import Path
from aiohttp import ClientSession
from typing import AsyncIterator, Optional

from ..typing import AsyncResult, Messages
from ..image import ImageResponse
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..cookies import get_cookies_dir

from .. import debug


class BlackboxCreateAgent(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.blackbox.ai"
    api_endpoints = {
        "llama-3.1-70b": "https://www.blackbox.ai/api/improve-prompt",
        "flux": "https://www.blackbox.ai/api/image-generator"
    }

    working = True
    supports_system_message = True
    supports_message_history = True

    default_model = 'llama-3.1-70b'
    chat_models = [default_model]
    image_models = ['flux']
    models = [*chat_models, *image_models]

    @classmethod
    def _get_cache_file(cls) -> Path:
        """Returns the path to the cache file."""
        dir = Path(get_cookies_dir())
        dir.mkdir(exist_ok=True)
        return dir / 'blackbox_create_agent.json'

    @classmethod
    def _load_cached_value(cls) -> str | None:
        cache_file = cls._get_cache_file()
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return data.get('validated_value')
            except Exception as e:
                debug.log(f"Error reading cache file: {e}")
        return None

    @classmethod
    def _save_cached_value(cls, value: str):
        cache_file = cls._get_cache_file()
        try:
            with open(cache_file, 'w') as f:
                json.dump({'validated_value': value}, f)
        except Exception as e:
            debug.log(f"Error writing to cache file: {e}")

    @classmethod
    async def fetch_validated(cls) -> Optional[str]:
        """
        Asynchronously retrieves the validated value from cache or website.

        :return: The validated value or None if retrieval fails.
        """
        cached_value = cls._load_cached_value()
        if cached_value:
            return cached_value

        js_file_pattern = r'static/chunks/\d{4}-[a-fA-F0-9]+\.js'
        v_pattern = r'j\s*=\s*[\'"]([0-9a-fA-F-]{36})[\'"]'

        def is_valid_context(text: str) -> bool:
            """Checks if the context is valid."""
            return any(char + '=' in text for char in 'abcdefghijklmnopqrstuvwxyz')

        async with ClientSession() as session:
            try:
                async with session.get(cls.url) as response:
                    if response.status != 200:
                        debug.log("Failed to download the page.")
                        return cached_value

                    page_content = await response.text()
                    js_files = re.findall(js_file_pattern, page_content)

                for js_file in js_files:
                    js_url = f"{cls.url}/_next/{js_file}"
                    async with session.get(js_url) as js_response:
                        if js_response.status == 200:
                            js_content = await js_response.text()
                            for match in re.finditer(v_pattern, js_content):
                                start = max(0, match.start() - 50)
                                end = min(len(js_content), match.end() + 50)
                                context = js_content[start:end]

                                if is_valid_context(context):
                                    validated_value = match.group(1)
                                    cls._save_cached_value(validated_value)
                                    return validated_value
            except Exception as e:
                debug.log(f"Error while retrieving validated_value: {e}")

        return cached_value

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        prompt: str = None,
        **kwargs
    ) -> AsyncIterator[str | ImageResponse]:
        """
        Creates an async generator for text or image generation.
        """
        if model in cls.chat_models:
            async for text in cls._generate_text(model, messages, proxy=proxy, **kwargs):
                yield text
        elif model in cls.image_models:
            prompt = messages[-1]['content']
            async for image in cls._generate_image(model, prompt, proxy=proxy, **kwargs):
                yield image
        else:
            raise ValueError(f"Model {model} not supported")

    @classmethod
    async def _generate_text(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        max_retries: int = 3,
        delay: int = 1,
        max_tokens: int = None,
        **kwargs
    ) -> AsyncIterator[str]:
        headers = cls._get_headers()

        for outer_attempt in range(2):  # Add outer loop for retrying with a new key
            validated_value = await cls.fetch_validated()
            if not validated_value:
                raise RuntimeError("Failed to get validated value")

            async with ClientSession(headers=headers) as session:
                api_endpoint = cls.api_endpoints[model]

                data = {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "validated": validated_value
                }

                for attempt in range(max_retries):
                    try:
                        async with session.post(api_endpoint, json=data, proxy=proxy) as response:
                            response.raise_for_status()
                            response_data = await response.json()

                            if response_data.get('status') == 200 and 'prompt' in response_data:
                                yield response_data['prompt']
                                return  # Successful execution
                            else:
                                raise KeyError("Invalid response format or missing 'prompt' key")
                    except Exception as e:
                        if attempt == max_retries - 1:
                            if outer_attempt == 0:  # If this is the first attempt with this key
                                # Remove the cached key and try to get a new one
                                cls._save_cached_value("")
                                debug.log("Invalid key, trying to get a new one...")
                                break  # Exit the inner loop to get a new key
                            else:
                                raise RuntimeError(f"Error after all attempts: {str(e)}")
                        else:
                            wait_time = delay * (2 ** attempt) + random.uniform(0, 1)
                            debug.log(f"Attempt {attempt + 1} failed. Retrying in {wait_time:.2f} seconds...")
                            await asyncio.sleep(wait_time)

    @classmethod
    async def _generate_image(
        cls,
        model: str,
        prompt: str,
        proxy: str = None,
        **kwargs
    ) -> AsyncIterator[ImageResponse]:
        headers = {
            **cls._get_headers()
        }

        api_endpoint = cls.api_endpoints[model]

        async with ClientSession(headers=headers) as session:
            data = {
                "query": prompt
            }

            async with session.post(api_endpoint, json=data, proxy=proxy) as response:
                response.raise_for_status()
                response_data = await response.json()

                if 'markdown' in response_data:
                    # Extract URL from markdown format: ![](url)
                    image_url = re.search(r'\!\[\]\((.*?)\)', response_data['markdown'])
                    if image_url:
                        yield ImageResponse(images=[image_url.group(1)], alt=prompt)
                    else:
                        raise ValueError("Could not extract image URL from markdown")
                else:
                    raise KeyError("'markdown' key not found in response")

    @staticmethod
    def _get_headers() -> dict:
        return {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'authorization': f'Bearer 56c8eeff9971269d7a7e625ff88e8a83a34a556003a5c87c289ebe9a3d8a3d2c',
            'content-type': 'application/json',
            'origin': 'https://www.blackbox.ai',
            'referer': 'https://www.blackbox.ai',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        """
        Creates an async response for the provider.

        Args:
            model: The model to use
            messages: The messages to process
            proxy: Optional proxy to use
            **kwargs: Additional arguments

        Returns:
            AsyncResult: The response from the provider
        """
        if model in cls.chat_models:
            async for text in cls._generate_text(model, messages, proxy=proxy, **kwargs):
                return text
        elif model in cls.image_models:
            prompt = messages[-1]['content']
            async for image in cls._generate_image(model, prompt, proxy=proxy, **kwargs):
                return image
        else:
            raise ValueError(f"Model {model} not supported")
