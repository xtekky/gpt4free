
from __future__ import annotations

import base64
import json
from aiohttp import ClientSession, BaseConnector

from ..typing import AsyncResult, Messages
from ..requests.raise_for_status import raise_for_status
from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from .helper import get_connector

class Ecosia(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://www.ecosia.org"
    working = True
    supports_gpt_35_turbo = True
    default_model = "gpt-3.5-turbo-0125"
    models = [default_model, "green"]
    model_aliases = {"gpt-3.5-turbo": default_model}

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        connector: BaseConnector = None,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        headers = {
            "authority": "api.ecosia.org",
            "accept": "*/*",
            "origin": cls.url,
            "referer": f"{cls.url}/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36",
        }
        async with ClientSession(headers=headers, connector=get_connector(connector, proxy)) as session:
            data = {
                "messages": base64.b64encode(json.dumps(messages).encode()).decode()
            }
            api_url = f"https://api.ecosia.org/v2/chat/?sp={'eco' if model == 'green' else 'productivity'}"
            async with session.post(api_url, json=data) as response:
                await raise_for_status(response)
                async for chunk in response.content.iter_any():
                    if chunk:
                        yield chunk.decode(errors="ignore")