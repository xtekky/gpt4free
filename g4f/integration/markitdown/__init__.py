import re
import sys
import io
from typing import List, Union, BinaryIO, Optional, Any
from markitdown import MarkItDown as BaseMarkItDown
from markitdown._stream_info import StreamInfo
from markitdown._base_converter import DocumentConverterResult

from markitdown._exceptions import (
    FileConversionException,
    UnsupportedFormatException,
    FailedConversionAttempt,
)

from ._audio_converter import AudioConverter
from ._image_converter import ImageConverter

class MarkItDown(BaseMarkItDown):
    """(In preview) An extremely simple text-based document reader, suitable for LLM use.
    This reader will convert common file-types or webpages to Markdown."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.register_converter(AudioConverter())
        self.register_converter(ImageConverter())

    def _convert(
        self, *, file_stream: BinaryIO, stream_info_guesses: List[StreamInfo], **kwargs
    ) -> DocumentConverterResult:
        res: Union[None, DocumentConverterResult] = None

        # Keep track of which converters throw exceptions
        failed_attempts: List[FailedConversionAttempt] = []

        # Create a copy of the page_converters list, sorted by priority.
        # We do this with each call to _convert because the priority of converters may change between calls.
        # The sort is guaranteed to be stable, so converters with the same priority will remain in the same order.
        sorted_registrations = sorted(self._converters, key=lambda x: x.priority)

        # Remember the initial stream position so that we can return to it
        cur_pos = file_stream.tell()

        for stream_info in stream_info_guesses + [StreamInfo()]:
            for converter_registration in sorted_registrations:
                converter = converter_registration.converter
                # Sanity check -- make sure the cur_pos is still the same
                assert (
                    cur_pos == file_stream.tell()
                ), f"File stream position should NOT change between guess iterations"

                _kwargs = {k: v for k, v in kwargs.items()}

                # Copy any additional global options
                if "llm_client" not in _kwargs and self._llm_client is not None:
                    _kwargs["llm_client"] = self._llm_client

                if "llm_model" not in _kwargs and self._llm_model is not None:
                    _kwargs["llm_model"] = self._llm_model

                if "style_map" not in _kwargs and self._style_map is not None:
                    _kwargs["style_map"] = self._style_map

                if "exiftool_path" not in _kwargs and self._exiftool_path is not None:
                    _kwargs["exiftool_path"] = self._exiftool_path

                # Add the list of converters for nested processing
                _kwargs["_parent_converters"] = self._converters

                # Add legaxy kwargs
                if stream_info is not None:
                    if stream_info.extension is not None:
                        _kwargs["file_extension"] = stream_info.extension

                    if stream_info.url is not None:
                        _kwargs["url"] = stream_info.url

                # Check if the converter will accept the file, and if so, try to convert it
                _accepts = False
                try:
                    _accepts = converter.accepts(file_stream, stream_info, **_kwargs)
                except NotImplementedError:
                    pass

                # accept() should not have changed the file stream position
                assert (
                    cur_pos == file_stream.tell()
                ), f"{type(converter).__name__}.accept() should NOT change the file_stream position"

                # Attempt the conversion
                if _accepts:
                    try:
                        res = converter.convert(file_stream, stream_info, **_kwargs)
                    except Exception:
                        failed_attempts.append(
                            FailedConversionAttempt(
                                converter=converter, exc_info=sys.exc_info()
                            )
                        )
                    finally:
                        file_stream.seek(cur_pos)

                if res is not None:
                    if isinstance(res.text_content, str):
                        # Normalize the content
                        res.text_content = "\n".join(
                            [line.rstrip() for line in re.split(r"\r?\n", res.text_content)]
                        )
                        res.text_content = re.sub(r"\n{3,}", "\n\n", res.text_content)
                    return res

        # If we got this far without success, report any exceptions
        if len(failed_attempts) > 0:
            raise FileConversionException(attempts=failed_attempts)

        # Nothing can handle it!
        raise UnsupportedFormatException(
            f"Could not convert stream to Markdown. No converter attempted a conversion, suggesting that the filetype is simply not supported."
        )

    def convert_stream(
        self,
        stream: BinaryIO,
        *,
        stream_info: Optional[StreamInfo] = None,
        file_extension: Optional[str] = None,  # Deprecated -- use stream_info
        url: Optional[str] = None,  # Deprecated -- use stream_info
        **kwargs: Any,
    ) -> DocumentConverterResult:
        guesses: List[StreamInfo] = []

        # Do we have anything on which to base a guess?
        base_guess = None
        if stream_info is not None or file_extension is not None or url is not None:
            # Start with a non-Null base guess
            if stream_info is None:
                base_guess = StreamInfo()
            else:
                base_guess = stream_info

            if file_extension is not None:
                # Deprecated -- use stream_info
                assert base_guess is not None  # for mypy
                base_guess = base_guess.copy_and_update(extension=file_extension)

            if url is not None:
                # Deprecated -- use stream_info
                assert base_guess is not None  # for mypy
                base_guess = base_guess.copy_and_update(url=url)

        # Check if we have a seekable stream. If not, load the entire stream into memory.
        if not hasattr(stream, "seekable") or not stream.seekable():
            buffer = io.BytesIO()
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                buffer.write(chunk)
            buffer.seek(0)
            stream = buffer

        # Add guesses based on stream content
        guesses = self._get_stream_info_guesses(
            file_stream=stream, base_guess=base_guess or StreamInfo()
        )
        return self._convert(file_stream=stream, stream_info_guesses=guesses, **kwargs)
