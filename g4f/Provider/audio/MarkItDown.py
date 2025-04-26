from __future__ import annotations

import os

try:
    from markitdown import MarkItDown as MaItDo, StreamInfo
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
        md = MaItDo()
        for file, filename in media:
            text = None
            try:
                text = md.convert(file, stream_info=StreamInfo(filename=filename) if filename else None).text_content
            except TypeError:
                copyfile = get_tempfile(file, filename)
                try:
                    text = md.convert(copyfile).text_content
                finally:
                    os.remove(copyfile)
            text = text.split("### Audio Transcript:\n")[-1]
            if text:
                yield text