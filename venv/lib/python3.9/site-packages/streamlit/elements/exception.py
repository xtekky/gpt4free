# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import traceback
from typing import TYPE_CHECKING, List, Optional, cast

from typing_extensions import Final

import streamlit
from streamlit.errors import (
    MarkdownFormattedException,
    StreamlitAPIException,
    StreamlitAPIWarning,
    StreamlitDeprecationWarning,
    UncaughtAppException,
)
from streamlit.logger import get_logger
from streamlit.proto.Exception_pb2 import Exception as ExceptionProto
from streamlit.runtime.metrics_util import gather_metrics

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

LOGGER: Final = get_logger(__name__)

# When client.showErrorDetails is False, we show a generic warning in the
# frontend when we encounter an uncaught app exception.
_GENERIC_UNCAUGHT_EXCEPTION_TEXT: Final = "This app has encountered an error. The original error message is redacted to prevent data leaks.  Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app)."

# Extract the streamlit package path. Make it absolute, resolve aliases, and
# ensure there's a trailing path separator
_STREAMLIT_DIR: Final = os.path.join(
    os.path.realpath(os.path.dirname(streamlit.__file__)), ""
)


class ExceptionMixin:
    @gather_metrics("exception")
    def exception(self, exception: BaseException) -> "DeltaGenerator":
        """Display an exception.

        Parameters
        ----------
        exception : Exception
            The exception to display.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> e = RuntimeError('This is an exception of type RuntimeError')
        >>> st.exception(e)

        """
        exception_proto = ExceptionProto()
        marshall(exception_proto, exception)
        return self.dg._enqueue("exception", exception_proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)


def marshall(exception_proto: ExceptionProto, exception: BaseException) -> None:
    """Marshalls an Exception.proto message.

    Parameters
    ----------
    exception_proto : Exception.proto
        The Exception protobuf to fill out

    exception : BaseException
        The exception whose data we're extracting
    """
    # If this is a StreamlitAPIException, we prune all Streamlit entries
    # from the exception's stack trace.
    is_api_exception = isinstance(exception, StreamlitAPIException)
    is_deprecation_exception = isinstance(exception, StreamlitDeprecationWarning)
    is_markdown_exception = isinstance(exception, MarkdownFormattedException)
    is_uncaught_app_exception = isinstance(exception, UncaughtAppException)

    stack_trace = (
        []
        if is_deprecation_exception
        else _get_stack_trace_str_list(
            exception, strip_streamlit_stack_entries=is_api_exception
        )
    )

    # Some exceptions (like UserHashError) have an alternate_name attribute so
    # we can pretend to the user that the exception is called something else.
    if getattr(exception, "alternate_name", None) is not None:
        exception_proto.type = getattr(exception, "alternate_name")
    else:
        exception_proto.type = type(exception).__name__

    exception_proto.stack_trace.extend(stack_trace)
    exception_proto.is_warning = isinstance(exception, Warning)

    try:
        if isinstance(exception, SyntaxError):
            # SyntaxErrors have additional fields (filename, text, lineno,
            # offset) that we can use for a nicely-formatted message telling
            # the user what to fix.
            exception_proto.message = _format_syntax_error_message(exception)
        else:
            exception_proto.message = str(exception).strip()
            exception_proto.message_is_markdown = is_markdown_exception

    except Exception as str_exception:
        # Sometimes the exception's __str__/__unicode__ method itself
        # raises an error.
        exception_proto.message = ""
        LOGGER.warning(
            """

Streamlit was unable to parse the data from an exception in the user's script.
This is usually due to a bug in the Exception object itself. Here is some info
about that Exception object, so you can report a bug to the original author:

Exception type:
  %(etype)s

Problem:
  %(str_exception)s

Traceback:
%(str_exception_tb)s

        """
            % {
                "etype": type(exception).__name__,
                "str_exception": str_exception,
                "str_exception_tb": "\n".join(_get_stack_trace_str_list(str_exception)),
            }
        )

    if is_uncaught_app_exception:
        uae = cast(UncaughtAppException, exception)
        exception_proto.message = _GENERIC_UNCAUGHT_EXCEPTION_TEXT
        type_str = str(type(uae.exc))
        exception_proto.type = type_str.replace("<class '", "").replace("'>", "")


def _format_syntax_error_message(exception: SyntaxError) -> str:
    """Returns a nicely formatted SyntaxError message that emulates
    what the Python interpreter outputs, e.g.:

    > File "raven.py", line 3
    >   st.write('Hello world!!'))
    >                            ^
    > SyntaxError: invalid syntax

    """
    if exception.text:
        if exception.offset is not None:
            caret_indent = " " * max(exception.offset - 1, 0)
        else:
            caret_indent = ""

        return (
            'File "%(filename)s", line %(lineno)s\n'
            "  %(text)s\n"
            "  %(caret_indent)s^\n"
            "%(errname)s: %(msg)s"
            % {
                "filename": exception.filename,
                "lineno": exception.lineno,
                "text": exception.text.rstrip(),
                "caret_indent": caret_indent,
                "errname": type(exception).__name__,
                "msg": exception.msg,
            }
        )
    # If a few edge cases, SyntaxErrors don't have all these nice fields. So we
    # have a fall back here.
    # Example edge case error message: encoding declaration in Unicode string
    return str(exception)


def _get_stack_trace_str_list(
    exception: BaseException, strip_streamlit_stack_entries: bool = False
) -> List[str]:
    """Get the stack trace for the given exception.

    Parameters
    ----------
    exception : BaseException
        The exception to extract the traceback from

    strip_streamlit_stack_entries : bool
        If True, all traceback entries that are in the Streamlit package
        will be removed from the list. We do this for exceptions that result
        from incorrect usage of Streamlit APIs, so that the user doesn't see
        a bunch of noise about ScriptRunner, DeltaGenerator, etc.

    Returns
    -------
    list
        The exception traceback as a list of strings

    """
    extracted_traceback: Optional[traceback.StackSummary] = None
    if isinstance(exception, StreamlitAPIWarning):
        extracted_traceback = exception.tacked_on_stack
    elif hasattr(exception, "__traceback__"):
        extracted_traceback = traceback.extract_tb(exception.__traceback__)

    if isinstance(exception, UncaughtAppException):
        extracted_traceback = traceback.extract_tb(exception.exc.__traceback__)

    # Format the extracted traceback and add it to the protobuf element.
    if extracted_traceback is None:
        stack_trace_str_list = [
            "Cannot extract the stack trace for this exception. "
            "Try calling exception() within the `catch` block."
        ]
    else:
        if strip_streamlit_stack_entries:
            extracted_frames = _get_nonstreamlit_traceback(extracted_traceback)
            stack_trace_str_list = traceback.format_list(extracted_frames)
        else:
            stack_trace_str_list = traceback.format_list(extracted_traceback)

    stack_trace_str_list = [item.strip() for item in stack_trace_str_list]

    return stack_trace_str_list


def _is_in_streamlit_package(file: str) -> bool:
    """True if the given file is part of the streamlit package."""
    try:
        common_prefix = os.path.commonprefix([os.path.realpath(file), _STREAMLIT_DIR])
    except ValueError:
        # Raised if paths are on different drives.
        return False

    return common_prefix == _STREAMLIT_DIR


def _get_nonstreamlit_traceback(
    extracted_tb: traceback.StackSummary,
) -> List[traceback.FrameSummary]:
    return [
        entry for entry in extracted_tb if not _is_in_streamlit_package(entry.filename)
    ]
