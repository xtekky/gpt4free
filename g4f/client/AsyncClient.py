from __future__ import annotations

from .Client import Client, Chat, Images, Completions
from .Client import async_iter_response, async_iter_append_model_and_provider
from aiohttp import ClientSession
from typing import Union, AsyncIterator

class AsyncClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat = AsyncChat(self)
        self._images = AsyncImages(self)

    @property
    def images(self) -> 'AsyncImages':
        return self._images

class AsyncCompletions(Completions):
    async def async_create(self, *args, **kwargs) -> Union['ChatCompletion', AsyncIterator['ChatCompletionChunk']]:
        response = await super().async_create(*args, **kwargs)
        async for result in response:
            return result

class AsyncChat(Chat):
    def __init__(self, client: AsyncClient):
        self.completions = AsyncCompletions(client)

class AsyncImages(Images):
    async def async_generate(self, *args, **kwargs) -> 'ImagesResponse':
        return await super().async_generate(*args, **kwargs)

    async def _fetch_image(self, url: str) -> bytes:
        async with ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    raise Exception(f"Failed to fetch image from {url}, status code {resp.status}")
