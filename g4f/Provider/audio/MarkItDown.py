from __future__ import annotations

import tempfile
import shutil
import os

try:
    from markitdown import MarkItDown as MaItDo, StreamInfo
    has_markitdown = True
except ImportError:
    has_markitdown = False

from ...typing import AsyncResult, Messages, MediaListType
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
            try:
                text = md.convert(file, stream_info=StreamInfo(filename=filename)).text_content
            except TypeError:
                # Copy SpooledTemporaryFile to a NamedTemporaryFile
                copyfile = tempfile.NamedTemporaryFile(suffix=filename, delete=False)
                shutil.copyfileobj(file, copyfile)
                copyfile.close()
                file.close()
                # Use the NamedTemporaryFile for conversion
                text = md.convert(copyfile.name, stream_info=StreamInfo(filename=filename)).text_content
                os.remove(copyfile.name)
            text = text.split("### Audio Transcript:\n")[-1]
            if text:
                yield text