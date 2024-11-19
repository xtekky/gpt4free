from __future__ import annotations

import json
import aiohttp
from ..typing import AsyncResult, Messages
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import format_prompt

class RobocodersAPI(AsyncGeneratorProvider, ProviderModelMixin):
    label = "API Robocoders AI"
    url = "https://api.robocoders.ai/docs"
    api_endpoint = "https://api.robocoders.ai/chat"
    working = True
    supports_message_history = True
    default_model = 'GeneralCodingAgent'
    agent = [default_model, "RepoAgent", "FrontEndAgent"]
    models = [*agent]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        async with aiohttp.ClientSession() as session:
            access_token = await cls._get_access_token(session)
            if not access_token:
                raise Exception("Failed to get access token")

            session_id = await cls._create_session(session, access_token)
            if not session_id:
                raise Exception("Failed to create session")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}"
            }
            
            prompt = format_prompt(messages)
            
            data = {
                "sid": session_id,
                "prompt": prompt,
                "agent": model
            }

            async with session.post(cls.api_endpoint, headers=headers, json=data, proxy=proxy) as response:
                if response.status != 200:
                    raise Exception(f"Error: {response.status}")
                
                async for line in response.content:
                    if line:
                        try:
                            response_data = json.loads(line)
                            message = response_data.get('message', '')
                            if message:
                                yield message
                        except json.JSONDecodeError:
                            pass

    @staticmethod
    async def _get_access_token(session: aiohttp.ClientSession) -> str:
        url_auth = 'https://api.robocoders.ai/auth'
        headers_auth = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-language': 'en-US,en;q=0.9',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
        }

        async with session.get(url_auth, headers=headers_auth) as response:
            if response.status == 200:
                text = await response.text()
                return text.split('id="token">')[1].split('</pre>')[0].strip()
        return None

    @staticmethod
    async def _create_session(session: aiohttp.ClientSession, access_token: str) -> str:
        url_create_session = 'https://api.robocoders.ai/create-session'
        headers_create_session = {
            'Authorization': f'Bearer {access_token}'
        }

        async with session.get(url_create_session, headers=headers_create_session) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('sid')
        return None

