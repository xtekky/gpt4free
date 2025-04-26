from __future__ import annotations

import json
from aiohttp import ClientSession, FormData

from ..typing import AsyncResult, Messages, MediaListType
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..requests.raise_for_status import raise_for_status
from ..image import to_bytes, is_accepted_format
from .helper import format_prompt

class Dynaspark(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://dynaspark.onrender.com"
    login_url = None
    api_endpoint = "https://dynaspark.onrender.com/generate_response"
    
    working = True
    needs_auth = False
    use_nodriver = True
    supports_stream = True
    supports_system_message = False
    supports_message_history = False

    default_model = 'gemini-1.5-flash'
    default_vision_model = default_model
    vision_models = [default_vision_model, 'gemini-1.5-flash-8b', 'gemini-2.0-flash', 'gemini-2.0-flash-lite']
    models = vision_models
    
    model_aliases = {
        "gemini-1.5-flash": "gemini-1.5-flash-8b",
        "gemini-2.0-flash": "gemini-2.0-flash-lite",
    }

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        media: MediaListType = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'origin': 'https://dynaspark.onrender.com',
            'referer': 'https://dynaspark.onrender.com/',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
            'x-requested-with': 'XMLHttpRequest'
        }
        async with ClientSession(headers=headers) as session:
            form = FormData()
            form.add_field('user_input', format_prompt(messages))
            form.add_field('ai_model', model)

            if media is not None and len(media) > 0:
                image, image_name = media[0]
                image_bytes = to_bytes(image)
                form.add_field('file', image_bytes, filename=image_name, content_type=is_accepted_format(image_bytes))

            async with session.post(f"{cls.api_endpoint}", data=form, proxy=proxy) as response:
                await raise_for_status(response)
                response_text = await response.text()
                response_json = json.loads(response_text)
                yield response_json["response"]
