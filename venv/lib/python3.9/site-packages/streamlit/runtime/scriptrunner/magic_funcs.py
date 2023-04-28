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

from typing import Any

from streamlit.runtime.metrics_util import gather_metrics


@gather_metrics("magic")
def transparent_write(*args: Any) -> Any:
    """The function that gets magic-ified into Streamlit apps.
    This is just st.write, but returns the arguments you passed to it.
    """
    import streamlit as st

    st.write(*args)
    if len(args) == 1:
        return args[0]
    return args
