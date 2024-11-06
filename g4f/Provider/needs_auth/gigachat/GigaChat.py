from __future__ import annotations

import os
import ssl
import time
import uuid

import json
from aiohttp import ClientSession, TCPConnector, BaseConnector
from g4f.requests import raise_for_status

from ....typing import AsyncResult, Messages
from ...base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ....errors import MissingAuthError
from ...helper import get_connector

access_token = ""
token_expires_at = 0

class GigaChat(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://developers.sber.ru/gigachat"
    working = True
    supports_message_history = True
    supports_system_message = True
    supports_stream = True
    needs_auth = True
    default_model = "GigaChat:latest"
    models = ["GigaChat:latest", "GigaChat-Plus", "GigaChat-Pro"]

    @classmethod
    async def create_async_generator(
            cls,
            model: str,
            messages: Messages,
            stream: bool = True,
            proxy: str = None,
            api_key: str = None,
            connector: BaseConnector = None,
            scope: str = "GIGACHAT_API_PERS",
            update_interval: float = 0,
            **kwargs
    ) -> AsyncResult:
        global access_token, token_expires_at
        model = cls.get_model(model)
        if not api_key:
            raise MissingAuthError('Missing "api_key"')
        
        cafile = os.path.join(os.path.dirname(__file__), "russian_trusted_root_ca_pem.crt")
        ssl_context = ssl.create_default_context(cafile=cafile) if os.path.exists(cafile) else None
        if connector is None and ssl_context is not None:
            connector = TCPConnector(ssl_context=ssl_context)
        async with ClientSession(connector=get_connector(connector, proxy)) as session:
            if token_expires_at - int(time.time() * 1000) < 60000:
                async with session.post(url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                                        headers={"Authorization": f"Bearer {api_key}",
                                                 "RqUID": str(uuid.uuid4()),
                                                 "Content-Type": "application/x-www-form-urlencoded"},
                                        data={"scope": scope}) as response:
                    await raise_for_status(response)
                    data = await response.json()
                access_token = data['access_token']
                token_expires_at = data['expires_at']

            async with session.post(url="https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                                    headers={"Authorization": f"Bearer {access_token}"},
                                    json={
                                        "model": model,
                                        "messages": messages,
                                        "stream": stream,
                                        "update_interval": update_interval,
                                        **kwargs
                                    }) as response:
                await raise_for_status(response)

                async for line in response.content:
                    if not stream:
                        yield json.loads(line.decode("utf-8"))['choices'][0]['message']['content']
                        return

                    if line and line.startswith(b"data:"):
                        line = line[6:-1]  # remove "data: " prefix and "\n" suffix
                        if line.strip() == b"[DONE]":
                            return
                        else:
                            msg = json.loads(line.decode("utf-8"))['choices'][0]
                            content = msg['delta']['content']

                            if content:
                                yield content

                            if 'finish_reason' in msg:
                                return
