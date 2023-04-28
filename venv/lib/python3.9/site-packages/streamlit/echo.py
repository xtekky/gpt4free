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

import contextlib
import re
import textwrap
import traceback
from typing import Iterable, List, Optional

from streamlit.runtime.metrics_util import gather_metrics

_SPACES_RE = re.compile("\\s*")
_EMPTY_LINE_RE = re.compile("\\s*\n")


@gather_metrics("echo")
@contextlib.contextmanager
def echo(code_location="above"):
    """Use in a `with` block to draw some code on the app, then execute it.

    Parameters
    ----------
    code_location : "above" or "below"
        Whether to show the echoed code before or after the results of the
        executed code block.

    Example
    -------
    >>> import streamlit as st
    >>>
    >>> with st.echo():
    >>>     st.write('This code will be printed')

    """
    from streamlit import code, empty, source_util, warning

    if code_location == "below":
        show_code = code
        show_warning = warning
    else:
        placeholder = empty()
        show_code = placeholder.code
        show_warning = placeholder.warning

    try:
        # Get stack frame *before* running the echoed code. The frame's
        # line number will point to the `st.echo` statement we're running.
        frame = traceback.extract_stack()[-3]
        filename, start_line = frame.filename, frame.lineno

        # Read the file containing the source code of the echoed statement.
        with source_util.open_python_file(filename) as source_file:
            source_lines = source_file.readlines()

        # Get the indent of the first line in the echo block, skipping over any
        # empty lines.
        initial_indent = _get_initial_indent(source_lines[start_line:])

        # Iterate over the remaining lines in the source file
        # until we find one that's indented less than the rest of the
        # block. That's our end line.
        #
        # Note that this is *not* a perfect strategy, because
        # de-denting is not guaranteed to signal "end of block". (A
        # triple-quoted string might be dedented but still in the
        # echo block, for example.)
        # TODO: rewrite this to parse the AST to get the *actual* end of the block.
        lines_to_display: List[str] = []
        for line in source_lines[start_line:]:
            indent = _get_indent(line)
            if indent is not None and indent < initial_indent:
                break
            lines_to_display.append(line)

        code_string = textwrap.dedent("".join(lines_to_display))

        # Run the echoed code...
        yield

        # And draw the code string to the app!
        show_code(code_string, "python")

    except FileNotFoundError as err:
        show_warning("Unable to display code. %s" % err)


def _get_initial_indent(lines: Iterable[str]) -> int:
    """Return the indent of the first non-empty line in the list.
    If all lines are empty, return 0.
    """
    for line in lines:
        indent = _get_indent(line)
        if indent is not None:
            return indent

    return 0


def _get_indent(line: str) -> Optional[int]:
    """Get the number of whitespaces at the beginning of the given line.
    If the line is empty, or if it contains just whitespace and a newline,
    return None.
    """
    if _EMPTY_LINE_RE.match(line) is not None:
        return None

    match = _SPACES_RE.match(line)
    return match.end() if match is not None else 0
