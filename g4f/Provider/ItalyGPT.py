from .base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ..typing import AsyncResult, Messages
from aiohttp import ClientSession

class ItalyGPT(AsyncGeneratorProvider, ProviderModelMixin):
    label = "ItalyGPT"
    url = "https://italygpt.it"
    working = True
    supports_system_message = True
    supports_message_history = True

    default_model = "gpt-4o"
    models = [default_model]

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        stream: bool = True,
        proxy: str = None,
        **kwargs
    ) -> AsyncResult:
        model = cls.get_model(model)
        headers = {
            "content-type": "application/json",
            "origin": "https://italygpt.it",
            "referer": "https://italygpt.it/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        payload = {
            "messages": messages,
            "stream": stream,
        }
        async with ClientSession() as session:
            async with session.post(
                f"{cls.url}/api/chat/",
                json=payload,
                headers=headers,
                proxy=proxy,
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.content:
                    if chunk:
                        yield chunk.decode()
