from __future__ import annotations

import os
import asyncio

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
                if isinstance(file, str) and file.startswith(("http://", "https://")):
                    result = md.convert_url(file)
                else:
                    result = md.convert(file, stream_info=StreamInfo(filename=filename) if filename else None)
                if asyncio.iscoroutine(result.text_content):
                    text = await result.text_content
                else:
                    text = result.text_content
            except TypeError:
                copyfile = get_tempfile(file, filename)
                try:
                    result = md.convert(copyfile)
                    if asyncio.iscoroutine(result.text_content):
                        text = await result.text_content
                    else:
                        text = result.text_content
                finally:
                    os.remove(copyfile)
            text = text.split("### Audio Transcript:\n")[-1]
            if text:
                yield text