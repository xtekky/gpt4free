from __future__ import annotations

import json

from ...typing import Messages, AsyncResult, MediaListType
from ...requests import StreamSession
from ...image import to_data_uri
from ...providers.base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...providers.response import RawResponse
from ... import debug

class BackendApi(AsyncGeneratorProvider, ProviderModelMixin):
    ssl = None
    headers = {}

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        media: MediaListType = None,
        api_key: str = None,
        **kwargs
    ) -> AsyncResult:
        debug.log(f"{cls.__name__}: {api_key}")
        if media is not None:
            for i in range(len(media)):
                media[i] = (to_data_uri(media[i][0], media[i][1]), media[i][1])
        async with StreamSession(
            headers={"Accept": "text/event-stream", **cls.headers},
        ) as session:
            async with session.post(f"{cls.url}/backend-api/v2/conversation", json={
                "model": model,
                "messages": messages,
                "media": media,
                "api_key": api_key,
                **kwargs
            }, ssl=cls.ssl) as response:
                async for line in response.iter_lines():
                    yield RawResponse(**json.loads(line))