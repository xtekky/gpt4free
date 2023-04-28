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

from typing import NoReturn

from streamlit.runtime.scriptrunner import (
    RerunData,
    RerunException,
    StopException,
    get_script_run_ctx,
)


def stop() -> NoReturn:
    """Stops execution immediately.

    Streamlit will not run any statements after `st.stop()`.
    We recommend rendering a message to explain why the script has stopped.
    When run outside of Streamlit, this will raise an Exception.

    Example
    -------
    >>> import streamlit as st
    >>>
    >>> name = st.text_input('Name')
    >>> if not name:
    >>>   st.warning('Please input a name.')
    >>>   st.stop()
    >>> st.success('Thank you for inputting a name.')

    """
    raise StopException()


def rerun() -> NoReturn:
    """Rerun the script immediately.

    When `st.experimental_rerun()` is called, the script is halted - no
    more statements will be run, and the script will be queued to re-run
    from the top.

    If this function is called outside of Streamlit, it will raise an
    Exception.
    """

    ctx = get_script_run_ctx()

    query_string = ""
    page_script_hash = ""
    if ctx is not None:
        query_string = ctx.query_string
        page_script_hash = ctx.page_script_hash

    raise RerunException(
        RerunData(
            query_string=query_string,
            page_script_hash=page_script_hash,
        )
    )
