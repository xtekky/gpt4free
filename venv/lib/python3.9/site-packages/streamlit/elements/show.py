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

import sys
from typing import Any

import streamlit as st
from streamlit import code_util, string_util
from streamlit.runtime.metrics_util import gather_metrics


@gather_metrics("experimental_show")
def show(*args: Any) -> None:
    """Write arguments and *argument names* to your app for debugging purposes.

    Show() has similar properties to write():

        1. You can pass in multiple arguments, all of which will be debugged.
        2. It returns None, so it's "slot" in the app cannot be reused.

    Note: This is an experimental feature. See
    https://docs.streamlit.io/library/advanced-features/prerelease#experimental for more information.

    Parameters
    ----------
    *args : any
        One or many objects to debug in the App.

    Example
    -------
    >>> import streamlit as st
    >>> import pandas as pd
    >>>
    >>> dataframe = pd.DataFrame({
    ...     'first column': [1, 2, 3, 4],
    ...     'second column': [10, 20, 30, 40],
    ... })
    >>> st.experimental_show(dataframe)

    Notes
    -----
    This is an experimental feature with usage limitations:

    - The method must be called with the name `show`.
    - Must be called in one line of code, and only once per line.
    - When passing multiple arguments the inclusion of `,` or `)` in a string
        argument may cause an error.

    """
    if not args:
        return

    try:
        import inspect

        # Get the calling line of code
        current_frame = inspect.currentframe()
        if current_frame is None:
            st.warning("`show` not enabled in the shell")
            return

        # Use two f_back because of telemetry decorator
        if current_frame.f_back is not None and current_frame.f_back.f_back is not None:
            lines = inspect.getframeinfo(current_frame.f_back.f_back)[3]
        else:
            lines = None

        if not lines:
            st.warning("`show` not enabled in the shell")
            return

        # Parse arguments from the line
        line = lines[0].split("show", 1)[1]
        inputs = code_util.get_method_args_from_code(args, line)

        # Escape markdown and add deltas
        for idx, input in enumerate(inputs):
            escaped = string_util.escape_markdown(input)

            st.markdown("**%s**" % escaped)
            st.write(args[idx])

    except Exception as raised_exc:
        _, exc, exc_tb = sys.exc_info()
        if exc is None:
            # Presumably, exc should never be None, but it is typed as
            # Optional, and I don't know the internals of sys.exc_info() well
            # enough to just use a cast here. Hence, the runtime check.
            raise RuntimeError(
                "Unexpected state: exc was None. If you see this message, "
                "please create an issue at "
                "https://github.com/streamlit/streamlit/issues"
            ) from raised_exc
        st.exception(exc)
