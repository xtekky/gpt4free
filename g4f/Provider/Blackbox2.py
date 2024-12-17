from __future__ import annotations

import random
import asyncio
import re
import json
from pathlib import Path
from aiohttp import ClientSession
from typing import AsyncIterator

from ..typing import AsyncResult, Messages
from ..image import ImageResponse
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..cookies import get_cookies_dir

from .. import debug

class Blackbox2(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.blackbox.ai"
    api_endpoints = {
        "llama-3.1-70b": "https://www.blackbox.ai/api/improve-prompt",
        "flux": "https://www.blackbox.ai/api/image-generator"
    }

    working = True
    supports_system_message = True
    supports_message_history = True
    supports_stream = False

    default_model = 'llama-3.1-70b'
    chat_models = ['llama-3.1-70b']
    image_models = ['flux']
    models = [*chat_models, *image_models]

    @classmethod
    def _get_cache_file(cls) -> Path:
        """Returns the path to the cache file."""
        dir = Path(get_cookies_dir())
        dir.mkdir(exist_ok=True)
        return dir / 'blackbox2.json'

    @classmethod
    def _load_cached_license(cls) -> str | None:
        """Loads the license key from the cache."""
        cache_file = cls._get_cache_file()
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return data.get('license_key')
            except Exception as e:
                debug.log(f"Error reading cache file: {e}")
        return None

    @classmethod
    def _save_cached_license(cls, license_key: str):
        """Saves the license key to the cache."""
        cache_file = cls._get_cache_file()
        try:
            with open(cache_file, 'w') as f:
                json.dump({'license_key': license_key}, f)
        except Exception as e:
            debug.log(f"Error writing to cache file: {e}")

    @classmethod
    async def _get_license_key(cls, session: ClientSession) -> str:
        cached_license = cls._load_cached_license()
        if cached_license:
            return cached_license

        try:
            async with session.get(cls.url) as response:
                html = await response.text()
                js_files = re.findall(r'static/chunks/\d{4}-[a-fA-F0-9]+\.js', html)

                license_format = r'["\'](\d{6}-\d{6}-\d{6}-\d{6}-\d{6})["\']'

                def is_valid_context(text_around):
                    return any(char + '=' in text_around for char in 'abcdefghijklmnopqrstuvwxyz')

                for js_file in js_files:
                    js_url = f"{cls.url}/_next/{js_file}"
                    async with session.get(js_url) as js_response:
                        js_content = await js_response.text()
                        for match in re.finditer(license_format, js_content):
                            start = max(0, match.start() - 10)
                            end = min(len(js_content), match.end() + 10)
                            context = js_content[start:end]

                            if is_valid_context(context):
                                license_key = match.group(1)
                                cls._save_cached_license(license_key)
                                return license_key

                raise ValueError("License key not found")
        except Exception as e:
            debug.log(f"Error getting license key: {str(e)}")
            raise

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        proxy: str = None,
        max_retries: int = 3,
        delay: int = 1,
        max_tokens: int = None,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = cls.default_model

        if model in cls.chat_models:
            async for result in cls._generate_text(model, messages, proxy, max_retries, delay, max_tokens):
                yield result
        elif model in cls.image_models:
            prompt = messages[-1]["content"]
            async for result in cls._generate_image(model, prompt, proxy):
                yield result
        else:
            raise ValueError(f"Unsupported model: {model}")

    @classmethod
    async def _generate_text(
        cls, 
        model: str, 
        messages: Messages, 
        proxy: str = None, 
        max_retries: int = 3, 
        delay: int = 1,
        max_tokens: int = None,
    ) -> AsyncIterator[str]:
        headers = cls._get_headers()

        async with ClientSession(headers=headers) as session:
            license_key = await cls._get_license_key(session)
            api_endpoint = cls.api_endpoints[model]

            data = {
                "messages": messages,
                "max_tokens": max_tokens,
                "validated": license_key
            }

            for attempt in range(max_retries):
                try:
                    async with session.post(api_endpoint, json=data, proxy=proxy) as response:
                        response.raise_for_status()
                        response_data = await response.json()
                        if 'prompt' in response_data:
                            yield response_data['prompt']
                            return
                        else:
                            raise KeyError("'prompt' key not found in the response")
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"Error after {max_retries} attempts: {str(e)}")
                    else:
                        wait_time = delay * (2 ** attempt) + random.uniform(0, 1)
                        debug.log(f"Attempt {attempt + 1} failed. Retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)

    @classmethod
    async def _generate_image(
        cls, 
        model: str, 
        prompt: str, 
        proxy: str = None
    ) -> AsyncIterator[ImageResponse]:
        headers = cls._get_headers()
        api_endpoint = cls.api_endpoints[model]

        async with ClientSession(headers=headers) as session:
            data = {
                "query": prompt
            }

            async with session.post(api_endpoint, headers=headers, json=data, proxy=proxy) as response:
                response.raise_for_status()
                response_data = await response.json()

                if 'markdown' in response_data:
                    image_url = response_data['markdown'].split('(')[1].split(')')[0]
                    yield ImageResponse(images=image_url, alt=prompt)

    @staticmethod
    def _get_headers() -> dict:
        return {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'text/plain;charset=UTF-8',
            'origin': 'https://www.blackbox.ai',
            'referer': 'https://www.blackbox.ai',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }
