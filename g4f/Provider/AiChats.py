from __future__ import annotations

import json
import base64
from aiohttp import ClientSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import ImageResponse
from .helper import format_prompt

class AiChats(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://ai-chats.org"
    api_endpoint = "https://ai-chats.org/chat/send2/"
    working = True
    supports_gpt_4 = True
    supports_message_history = True
    default_model = 'gpt-4'
    models = ['gpt-4', 'dalle']

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        headers = {
            "accept": "application/json, text/event-stream",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "origin": cls.url,
            "pragma": "no-cache",
            "referer": f"{cls.url}/{'image' if model == 'dalle' else 'chat'}/",
            "sec-ch-ua": '"Chromium";v="127", "Not)A;Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            'cookie': 'muVyak=LSFNvUWqdgKkGprbDBsfieIoEMzjOQ; LSFNvUWqdgKkGprbDBsfieIoEMzjOQ=ac28831b98143847e83dbe004404e619-1725548624-1725548621; muVyak_hits=9; ai-chat-front=9d714d5dc46a6b47607c9a55e7d12a95; _csrf-front=76c23dc0a013e5d1e21baad2e6ba2b5fdab8d3d8a1d1281aa292353f8147b057a%3A2%3A%7Bi%3A0%3Bs%3A11%3A%22_csrf-front%22%3Bi%3A1%3Bs%3A32%3A%22K9lz0ezsNPMNnfpd_8gT5yEeh-55-cch%22%3B%7D',
        }

        async with ClientSession(headers=headers) as session:
            if model == 'dalle':
                prompt = messages[-1]['content'] if messages else ""
            else:
                prompt = format_prompt(messages)

            data = {
                "type": "image" if model == 'dalle' else "chat",
                "messagesHistory": [
                    {
                        "from": "you",
                        "content": prompt
                    }
                ]
            }

            try:
                async with session.post(cls.api_endpoint, json=data, proxy=proxy) as response:
                    response.raise_for_status()

                    if model == 'dalle':
                        response_json = await response.json()

                        if 'data' in response_json and response_json['data']:
                            image_url = response_json['data'][0].get('url')
                            if image_url:
                                async with session.get(image_url) as img_response:
                                    img_response.raise_for_status()
                                    image_data = await img_response.read()

                                base64_image = base64.b64encode(image_data).decode('utf-8')
                                base64_url = f"data:image/png;base64,{base64_image}"
                                yield ImageResponse(base64_url, prompt)
                            else:
                                yield f"Error: No image URL found in the response. Full response: {response_json}"
                        else:
                            yield f"Error: Unexpected response format. Full response: {response_json}"
                    else:
                        full_response = await response.text()
                        message = ""
                        for line in full_response.split('\n'):
                            if line.startswith('data: ') and line != 'data: ':
                                message += line[6:]

                        message = message.strip()
                        yield message
            except Exception as e:
                yield f"Error occurred: {str(e)}"

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> str:
        async for response in cls.create_async_generator(model, messages, proxy, **kwargs):
            if isinstance(response, ImageResponse):
                return response.images[0]
            return response
