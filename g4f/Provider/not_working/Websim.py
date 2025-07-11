from __future__ import annotations

import json
import random
import string
import asyncio
from aiohttp import ClientSession

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...requests.raise_for_status import raise_for_status
from ...errors import ResponseStatusError
from ...providers.response import ImageResponse
from ..helper import format_prompt, format_media_prompt


class Websim(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://websim.ai"
    login_url = None
    chat_api_endpoint = "https://websim.ai/api/v1/inference/run_chat_completion"
    image_api_endpoint = "https://websim.ai/api/v1/inference/run_image_generation"
    
    working = False
    needs_auth = False
    use_nodriver = False
    supports_stream = False
    supports_system_message = True
    supports_message_history = True

    default_model = 'gemini-2.5-pro'
    default_image_model = 'flux'
    image_models = [default_image_model]
    models = [default_model, 'gemini-2.5-flash'] + image_models

    @staticmethod
    def generate_project_id(for_image=False):
        """
        Generate a project ID in the appropriate format
        
        For chat: format like 'ke3_xh5gai3gjkmruomu'
        For image: format like 'kx0m131_rzz66qb2xoy7'
        """
        chars = string.ascii_lowercase + string.digits
        
        if for_image:
            first_part = ''.join(random.choices(chars, k=7))
            second_part = ''.join(random.choices(chars, k=12))
            return f"{first_part}_{second_part}"
        else:
            prefix = ''.join(random.choices(chars, k=3))
            suffix = ''.join(random.choices(chars, k=15))
            return f"{prefix}_{suffix}"

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        prompt: str = None,
        proxy: str = None,
        aspect_ratio: str = "1:1",
        project_id: str = None,
        **kwargs
    ) -> AsyncResult:
        is_image_request = model in cls.image_models
        
        if project_id is None:
            project_id = cls.generate_project_id(for_image=is_image_request)
        
        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'text/plain;charset=UTF-8',
            'origin': 'https://websim.ai',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
            'websim-flags;': ''
        }
        
        if is_image_request:
            headers['referer'] = 'https://websim.ai/@ISWEARIAMNOTADDICTEDTOPILLOW/ai-image-prompt-generator'
            async for result in cls._handle_image_request(
                project_id=project_id,
                messages=messages,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                headers=headers,
                proxy=proxy,
                **kwargs
            ):
                yield result
        else:
            headers['referer'] = 'https://websim.ai/@ISWEARIAMNOTADDICTEDTOPILLOW/zelos-ai-assistant'
            async for result in cls._handle_chat_request(
                project_id=project_id,
                messages=messages,
                headers=headers,
                proxy=proxy,
                **kwargs
            ):
                yield result

    @classmethod
    async def _handle_image_request(
        cls,
        project_id: str,
        messages: Messages,
        prompt: str,
        aspect_ratio: str,
        headers: dict,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        used_prompt = format_media_prompt(messages, prompt)
        
        async with ClientSession(headers=headers) as session:
            data = {
                "project_id": project_id,
                "prompt": used_prompt,
                "aspect_ratio": aspect_ratio
            }
            async with session.post(f"{cls.image_api_endpoint}", json=data, proxy=proxy) as response:
                await raise_for_status(response)
                response_text = await response.text()
                response_json = json.loads(response_text)
                image_url = response_json.get("url")
                if image_url:
                    yield ImageResponse(urls=[image_url], alt=used_prompt)

    @classmethod
    async def _handle_chat_request(
        cls,
        project_id: str,
        messages: Messages,
        headers: dict,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                async with ClientSession(headers=headers) as session:
                    data = {
                        "project_id": project_id,
                        "messages": messages
                    }
                    async with session.post(f"{cls.chat_api_endpoint}", json=data, proxy=proxy) as response:
                        if response.status == 429:
                            response_text = await response.text()
                            last_error = ResponseStatusError(f"Response {response.status}: {response_text}")
                            retry_count += 1
                            if retry_count < max_retries:
                                wait_time = 2 ** retry_count
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                raise last_error
                        
                        await raise_for_status(response)
                        
                        response_text = await response.text()
                        try:
                            response_json = json.loads(response_text)
                            content = response_json.get("content", "")
                            yield content.strip()
                            break
                        except json.JSONDecodeError:
                            yield response_text
                            break
                            
            except ResponseStatusError as e:
                if "Rate limit exceeded" in str(e) and retry_count < max_retries:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    await asyncio.sleep(wait_time)
                else:
                    if retry_count >= max_retries:
                        raise e
                    else:
                        raise
            except Exception as e:
                raise
