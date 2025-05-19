from __future__ import annotations

import os
import asyncio
from typing import Any

try:
    from ...integration.markitdown import MarkItDown as MaItDo, StreamInfo
    has_markitdown = True
except ImportError:
    has_markitdown = False

from ...typing import AsyncResult, Messages, MediaListType
from ...tools.files import get_tempfile
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin

class MarkItDown(AsyncGeneratorProvider, ProviderModelMixin):
    working = has_markitdown

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        media: MediaListType = None,
        llm_client: Any = None,
        **kwargs
    ) -> AsyncResult:
        if media is None:
            raise ValueError("MarkItDown requires media to be provided.")
        if not has_markitdown:
            raise ImportError("MarkItDown is not installed. Please install it with `pip install markitdown`.")
        md = MaItDo()
        for file, filename in media:
            text = None
            try:
                result = md.convert(
                    file,
                    stream_info=StreamInfo(filename=filename) if filename else None,
                    llm_client=llm_client,
                    llm_model=model
                )
                if asyncio.iscoroutine(result.text_content):
                    text = await result.text_content
                else:
                    text = result.text_content
            except TypeError:
                copyfile = get_tempfile(file, filename)
                try:
                    result = md.convert(
                        copyfile, 
                        llm_client=llm_client,
                        llm_model=model
                    )
                    if asyncio.iscoroutine(result.text_content):
                        text = await result.text_content
                    else:
                        text = result.text_content
                finally:
                    os.remove(copyfile)
            text = text.split("### Audio Transcript:\n")[-1]
            if text:
                yield text