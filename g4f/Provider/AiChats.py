from __future__ import annotations

import json
import base64
from aiohttp import ClientSession
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..image import ImageResponse

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
        }
        
        async with ClientSession(headers=headers) as session:
            prompt = cls.format_prompt(messages)
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
    def format_prompt(cls, messages: Messages) -> str:
        return messages[-1]['content'] if messages else ""
